"""
drone_demo.py  Pygame heuristic visualisation
================================================
Runs the DroneDataRelayEnv with a priority-based heuristic agent.

Vendored from github.com/FyneappleJuice/hermes_rl at commit a8a453f; the only
change is package imports. Run from the repository root with:
    python -m experiments.sim.drone_env.drone_demo

Requirements:
    pip install pygame numpy

Controls:
    SPACE   pause / resume
    R       reset episode
    +/-     speed up / slow down
    Q/ESC   quit
"""

import math
import sys
import time
import numpy as np
import pygame

from .drone_env import DroneDataRelayEnv, EnvConfig, JobConfig, SensorConfig

#  palette 
BG          = (15,  17,  26)
GRID        = (28,  32,  48)
WHITE       = (220, 220, 230)
MUTED       = (100, 105, 130)
BS_COL      = (140, 110, 255)
BS_RING     = (80,   60, 180)
WP_COL      = (50,  200, 160)
WP_RING     = (30,  140, 110)
SENSOR_COL  = (255, 190,  50)
SENSOR_BUSY = (80,  200, 255)
SENSOR_DONE = (80,   85, 100)
DRONE_COL   = (255,  70,  90)
DRONE_MOVE  = (255, 200,  60)
BEAM_COLS   = [(100, 160, 255), (80, 220, 130), (255, 160, 60)]
JOB_COLS    = [(100, 200, 255), (130, 255, 140), (255, 180, 80)]
PANEL_BG    = (22,  25,  38)
PANEL_EDGE  = (45,  50,  75)
GREEN       = (80,  220, 100)
RED         = (255,  80,  80)
ORANGE      = (255, 160,  50)
CYAN        = (80,  200, 255)

#  layout 
WIN_W, WIN_H = 1100, 700
CANVAS_X     = 10
CANVAS_Y     = 10
CANVAS_W     = 680
CANVAS_H     = 680
PANEL_X      = CANVAS_X + CANVAS_W + 10
PANEL_W      = WIN_W - PANEL_X - 10
PAD          = 16


def to_px(wx, wy, cfg):
    px = CANVAS_X + int(wx / cfg.plane_w * CANVAS_W)
    py = CANVAS_Y + int((1 - wy / cfg.plane_h) * CANVAS_H)
    return px, py


def radius_px(r, cfg):
    return int(r / cfg.plane_w * CANVAS_W)



class HeuristicAgent:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _arrival_time(self, from_wp, to_wp):
        return self.env.t + self.env._transit_time(from_wp, to_wp)

    def _rate(self, bs_idx, ch_idx, wp_idx, t):
        wp_pos = self.cfg.waypoints[wp_idx]
        bs     = self.cfg.base_stations[bs_idx]
        d      = self._dist(wp_pos, bs.position)
        return 0.0 if d > bs.radius else self.env._upload_rate(bs_idx, ch_idx, d, t)

    def _best_channel(self, bs_idx, wp_idx, t):
        best_ch, best_r = 0, -1.0
        for ci in range(self.env.n_channels):
            r = self._rate(bs_idx, ci, wp_idx, t)
            if r > best_r:
                best_r = r; best_ch = ci
        return best_ch, best_r

    def _best_upload_action(self):
        env = self.env; cfg = self.cfg
        best_act = None; best_score = -1.0
        for wi in range(env.n_waypoints):
            wp_pos = cfg.waypoints[wi]
            arr_t  = self._arrival_time(env.current_wp, wi)
            travel = env._transit_time(env.current_wp, wi)
            for bi in range(env.n_bs):
                d = self._dist(wp_pos, cfg.base_stations[bi].position)
                if d > cfg.base_stations[bi].radius:
                    continue
                ch, r = self._best_channel(bi, wi, arr_t)
                if r <= 0:
                    continue
                score = r / (1.0 + 0.05 * travel)
                if score > best_score:
                    best_score = score
                    best_act   = wi * env.n_bs * env.n_channels + bi * env.n_channels + ch
        return best_act if best_act is not None else (1 * env.n_bs * env.n_channels + 1 * env.n_channels)

    def _collect_action(self, ji):
        env = self.env; cfg = self.cfg
        sen = cfg.sensors[cfg.jobs[ji].sensor_idx]
        def wp_score(w):
            d = self._dist(cfg.waypoints[w], sen.position)
            return (0 if d <= sen.radius else 1, d)
        best_wp = min(range(env.n_waypoints), key=wp_score)
        arr_t   = self._arrival_time(env.current_wp, best_wp)
        best_bs, best_ch, best_r = 0, 0, -1.0
        for bi in range(env.n_bs):
            ch, r = self._best_channel(bi, best_wp, arr_t)
            if r > best_r:
                best_r = r; best_bs = bi; best_ch = ch
        return best_wp * env.n_bs * env.n_channels + best_bs * env.n_channels + best_ch

    def act(self):
        env = self.env; cfg = self.cfg
        if env.is_moving:
            return env.target_wp * env.n_bs * env.n_channels + env.active_bs * env.n_channels + env.active_ch

        pending = [ji for ji in range(env.n_jobs)
                   if env.job_collected[ji] and not env.job_done[ji] and not env.job_failed[ji]]
        uncollected = [ji for ji in range(env.n_jobs)
                       if not env.job_collected[ji] and not env.job_done[ji] and not env.job_failed[ji]]

        if pending:
            wp_pos = cfg.waypoints[env.current_wp]
            cur_best_r = 0.0; cur_best_bs = 0; cur_best_ch = 0
            for bi in range(env.n_bs):
                d = self._dist(wp_pos, cfg.base_stations[bi].position)
                if d <= cfg.base_stations[bi].radius:
                    ch, r = self._best_channel(bi, env.current_wp, env.t)
                    if r > cur_best_r:
                        cur_best_r = r; cur_best_bs = bi; cur_best_ch = ch
            if cur_best_r > 0:
                return env.current_wp * env.n_bs * env.n_channels + cur_best_bs * env.n_channels + cur_best_ch
            return self._best_upload_action()

        if uncollected:
            ji = min(uncollected, key=lambda j: env.job_deadlines[j])
            return self._collect_action(ji)

        return env.current_wp * env.n_bs * env.n_channels


#  drawing helpers 

def draw_dashed_circle(surf, col, centre, radius, dash=8, gap=6, width=1):
    total = 2 * math.pi * radius
    n     = max(1, int(total / (dash + gap)))
    step  = 2 * math.pi / n
    for i in range(n):
        a0  = i * step
        pts = [(centre[0] + radius * math.cos(t),
                centre[1] + radius * math.sin(t))
               for t in np.linspace(a0, a0 + step * dash / (dash + gap), 8)]
        if len(pts) >= 2:
            pygame.draw.lines(surf, col, False, pts, width)


def draw_sine_strip(surf, font_tiny, env, x, y, w, h):
    cfg = env.cfg; t = env.t
    for bi in range(env.n_bs):
        bx = x + bi * (w // 3); bw = w // 3 - 4
        pygame.draw.rect(surf, PANEL_BG,   (bx, y, bw, h), border_radius=4)
        pygame.draw.rect(surf, PANEL_EDGE, (bx, y, bw, h), 1, border_radius=4)
        surf.blit(font_tiny.render(f"BS{bi+1}", True, MUTED), (bx+4, y+2))
        for ci in range(3):
            phase = cfg.base_stations[bi].phase_offsets[ci]
            pts   = []
            for xi in range(bw - 2):
                v    = cfg.alpha * math.sin(cfg.omega * (t + xi * 0.4) + phase)
                norm = (v + cfg.alpha) / (2 * cfg.alpha)
                pts.append((bx + 1 + xi, y + h - 4 - int(norm * (h - 14))))
            if len(pts) >= 2:
                active = (bi == env.active_bs and ci == env.active_ch)
                pygame.draw.lines(surf, BEAM_COLS[ci], False, pts, 2 if active else 1)
            cur  = cfg.alpha * math.sin(cfg.omega * t + phase)
            norm = (cur + cfg.alpha) / (2 * cfg.alpha)
            pygame.draw.circle(surf, BEAM_COLS[ci],
                                (bx+1, y+h-4-int(norm*(h-14))), 3)


def draw_panel(surf, fonts, env, fps, speed):
    font, font_sm, font_tiny = fonts
    cfg = env.cfg
    x   = PANEL_X; y = CANVAS_Y

    pygame.draw.rect(surf, PANEL_BG,   (x, y, PANEL_W, WIN_H-20), border_radius=8)
    pygame.draw.rect(surf, PANEL_EDGE, (x, y, PANEL_W, WIN_H-20), 1, border_radius=8)
    x += PAD; y += PAD

    surf.blit(font.render("Drone Data Relay", True, WHITE), (x, y));     y += 28
    surf.blit(font_tiny.render("Agent Demo", True, MUTED), (x, y)); y += 22
    pygame.draw.line(surf, PANEL_EDGE, (x, y), (x+PANEL_W-2*PAD, y));   y += 10

    wp_pos  = cfg.waypoints[env.current_wp]
    bs_cfg  = cfg.base_stations[env.active_bs]
    d_to_bs = math.hypot(wp_pos[0]-bs_cfg.position[0], wp_pos[1]-bs_cfg.position[1])
    uploading = not env.is_moving and d_to_bs <= bs_cfg.radius
    cur_rate  = env._upload_rate(env.active_bs, env.active_ch, d_to_bs, env.t) if uploading else 0.0

    dl_info = ""
    if env.active_sensor >= 0:
        ji     = cfg.sensors[env.active_sensor].job_idx
        dl_info = f"S{env.active_sensor+1}  {env.collect_progress:.2f}/{cfg.jobs[ji].total_data:.1f} MB"

    upload_str = f"BS{env.active_bs+1} A· Ch{env.active_ch+1}" if uploading else ""
    rate_str   = f"{cur_rate:.4f} MB/s" if uploading else ""

    rows = [
        ("Timestep", f"{int(env.t)}",                                     WHITE),
        ("Step",     f"{env.step_count}",                                  WHITE),
        ("FPS",      f"{fps:.0f}",                                         WHITE),
        ("Speed",    f"{speed}x",                                          WHITE),
        ("Waypoint", f"W{env.current_wp+1}",                               WHITE),
        ("Status",   "MOVING" if env.is_moving else "IDLE",
                      DRONE_MOVE if env.is_moving else WHITE),
        ("Download", dl_info,   CYAN if env.active_sensor >= 0 else MUTED),
        ("Upload", upload_str, BEAM_COLS[env.active_ch] if uploading else MUTED),
        ("Rate",     rate_str,   GREEN if cur_rate > 0 else MUTED),
    ]
    for lbl, val, col in rows:
        surf.blit(font_tiny.render(lbl, True, MUTED), (x,       y))
        surf.blit(font_tiny.render(val, True, col),   (x+110,   y))
        y += 16

    y += 6
    pygame.draw.line(surf, PANEL_EDGE, (x, y), (x+PANEL_W-2*PAD, y)); y += 10

    surf.blit(font_sm.render("Jobs", True, WHITE), (x, y)); y += 18
    bar_w = PANEL_W - 2*PAD - 10

    for ji, job in enumerate(cfg.jobs):
        col    = JOB_COLS[ji % len(JOB_COLS)]
        rem    = env.job_data_remaining[ji]
        pct    = max(0.0, 1.0 - rem / job.total_data)
        dl_rem = max(0, int(job.deadline - env.t))
        s_idx  = job.sensor_idx
        dl_pct = 1.0 if env.job_collected[ji] else (
            min(1.0, env.collect_progress / job.total_data)
            if env.active_sensor == s_idx else 0.0)

        if env.job_done[ji]:               tag, tc = "DONE",    GREEN
        elif env.job_failed[ji]:           tag, tc = "MISSED",  RED
        elif env.job_collected[ji]:        tag, tc = "UPLOAD",  ORANGE
        elif env.active_sensor == s_idx:   tag, tc = "DL...",   CYAN
        else:                              tag, tc = "COLLECT",  col

        surf.blit(font_tiny.render(
            f"Job {ji}  [{tag}]  {rem:.2f}/{job.total_data:.1f} MB  dl:{dl_rem}",
            True, tc), (x, y)); y += 14

        # download progress bar (cyan, thin)
        pygame.draw.rect(surf, GRID, (x, y, bar_w, 4), border_radius=2)
        if dl_pct > 0:
            pygame.draw.rect(surf, CYAN, (x, y, int(bar_w*dl_pct), 4), border_radius=2)
        y += 6

        # upload progress bar
        pygame.draw.rect(surf, GRID, (x, y, bar_w, 7), border_radius=3)
        if pct > 0:
            fc = GREEN if env.job_done[ji] else (RED if env.job_failed[ji] else col)
            pygame.draw.rect(surf, fc, (x, y, int(bar_w*pct), 7), border_radius=3)
        pygame.draw.rect(surf, PANEL_EDGE, (x, y, bar_w, 7), 1, border_radius=3)
        y += 13

    y += 4
    pygame.draw.line(surf, PANEL_EDGE, (x, y), (x+PANEL_W-2*PAD, y)); y += 10

    surf.blit(font_sm.render("Upload rates (MB/s)", True, WHITE), (x, y)); y += 18
    cell_w = bar_w // (env.n_bs * env.n_channels)
    for bi in range(env.n_bs):
        for ci in range(env.n_channels):
            cx     = x + (bi*env.n_channels+ci)*cell_w
            bs_pos = cfg.base_stations[bi].position
            d      = math.hypot(wp_pos[0]-bs_pos[0], wp_pos[1]-bs_pos[1])
            in_rng = d <= cfg.base_stations[bi].radius
            r      = env._upload_rate(bi, ci, d, env.t) if in_rng else 0.0
            active = (bi == env.active_bs and ci == env.active_ch and uploading)
            pygame.draw.rect(surf, (40,50,80) if active else GRID,
                             (cx, y, cell_w-2, 34), border_radius=4)
            if active:
                pygame.draw.rect(surf, BEAM_COLS[ci], (cx, y, cell_w-2, 34), 1, border_radius=4)
            surf.blit(font_tiny.render(f"B{bi+1}C{ci+1}", True, MUTED), (cx+3, y+2))
            surf.blit(font_sm.render(f"{r:.3f}", True, BEAM_COLS[ci] if in_rng else MUTED),
                      (cx+3, y+16))
    y += 44

    pygame.draw.line(surf, PANEL_EDGE, (x, y), (x+PANEL_W-2*PAD, y)); y += 10
    surf.blit(font_sm.render("Channel waveforms", True, WHITE), (x, y)); y += 18
    draw_sine_strip(surf, font_tiny, env, x, y, bar_w, 60)
    y += 70

    pygame.draw.line(surf, PANEL_EDGE, (x, y), (x+PANEL_W-2*PAD, y)); y += 10
    for c in ["SPACE  pause / resume", "R      reset episode",
              "+/-    speed up / down", "Q/ESC  quit"]:
        surf.blit(font_tiny.render(c, True, MUTED), (x, y)); y += 15


def draw_canvas(surf, env, drone_px, fonts):
    _, font_sm, font_tiny = fonts
    cfg    = env.cfg
    wp_pos = cfg.waypoints[env.current_wp]

    pygame.draw.rect(surf, BG, (CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H))

    # grid
    for i in range(0, 101, 10):
        xi, _ = to_px(i, 0, cfg); _, yi = to_px(0, i, cfg)
        pygame.draw.line(surf, GRID, (xi, CANVAS_Y), (xi, CANVAS_Y+CANVAS_H), 1)
        pygame.draw.line(surf, GRID, (CANVAS_X, yi), (CANVAS_X+CANVAS_W, yi), 1)

    # BS radius rings
    for bi, bs in enumerate(cfg.base_stations):
        cx, cy = to_px(bs.position[0], bs.position[1], cfg)
        r_px   = radius_px(bs.radius, cfg)
        d      = math.hypot(wp_pos[0]-bs.position[0], wp_pos[1]-bs.position[1])
        active = (bi == env.active_bs and d <= bs.radius and not env.is_moving)
        col    = BEAM_COLS[env.active_ch] if active else BS_RING
        ring   = pygame.Surface((r_px*2+2, r_px*2+2), pygame.SRCALPHA)
        pygame.draw.circle(ring, (*col, 18), (r_px+1, r_px+1), r_px)
        surf.blit(ring, (cx-r_px-1, cy-r_px-1))
        draw_dashed_circle(surf, col, (cx, cy), r_px, width=1)

    # sensor radius rings
    for si, sen in enumerate(cfg.sensors):
        ji   = sen.job_idx
        busy = (env.active_sensor == si)
        col  = SENSOR_DONE if env.job_collected[ji] else (SENSOR_BUSY if busy else SENSOR_COL)
        cx, cy = to_px(sen.position[0], sen.position[1], cfg)
        r_px   = radius_px(sen.radius, cfg)
        ring   = pygame.Surface((r_px*2+2, r_px*2+2), pygame.SRCALPHA)
        pygame.draw.circle(ring, (*col, 25), (r_px+1, r_px+1), r_px)
        surf.blit(ring, (cx-r_px-1, cy-r_px-1))
        draw_dashed_circle(surf, col, (cx, cy), r_px, width=1)

    # transit path
    if env.is_moving:
        p1 = to_px(cfg.waypoints[env.current_wp][0], cfg.waypoints[env.current_wp][1], cfg)
        p2 = to_px(cfg.waypoints[env.target_wp][0],  cfg.waypoints[env.target_wp][1],  cfg)
        ddx, ddy = p2[0]-p1[0], p2[1]-p1[1]
        length = math.hypot(ddx, ddy)
        if length > 0:
            steps = max(1, int(length/10))
            for i in range(0, steps, 2):
                t0 = i/steps; t1 = min((i+1)/steps, 1.0)
                pygame.draw.line(surf, (80, 120, 200),
                    (int(p1[0]+ddx*t0), int(p1[1]+ddy*t0)),
                    (int(p1[0]+ddx*t1), int(p1[1]+ddy*t1)), 1)

    # download beam (drone ↔ sensor, cyan)
    if env.active_sensor >= 0 and not env.is_moving:
        sen    = cfg.sensors[env.active_sensor]
        sx, sy = to_px(sen.position[0], sen.position[1], cfg)
        dl_pct = min(1.0, env.collect_progress / max(0.01, cfg.jobs[sen.job_idx].total_data))
        beam   = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
        pygame.draw.line(beam, (*SENSOR_BUSY, int(80+dl_pct*150)),
                         (drone_px[0]-CANVAS_X, drone_px[1]-CANVAS_Y),
                         (sx-CANVAS_X, sy-CANVAS_Y), 2)
        surf.blit(beam, (CANVAS_X, CANVAS_Y))

    # upload beam (drone → BS, coloured by channel)
    if not env.is_moving:
        bs  = cfg.base_stations[env.active_bs]
        d   = math.hypot(wp_pos[0]-bs.position[0], wp_pos[1]-bs.position[1])
        if d <= bs.radius:
            r = env._upload_rate(env.active_bs, env.active_ch, d, env.t)
            if r > 0:
                bx, by = to_px(bs.position[0], bs.position[1], cfg)
                col    = BEAM_COLS[env.active_ch]
                alpha  = min(255, int(80 + r/(cfg.alpha+cfg.beta)*160))
                beam   = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
                pygame.draw.line(beam, (*col, alpha),
                                 (drone_px[0]-CANVAS_X, drone_px[1]-CANVAS_Y),
                                 (bx-CANVAS_X, by-CANVAS_Y),
                                 max(1, int(1 + r*3)))
                surf.blit(beam, (CANVAS_X, CANVAS_Y))

                # label at midpoint of beam
                label_txt = f"BS{env.active_bs+1} A· Ch{env.active_ch+1}   {r:.4f} MB/s"
                lsurf = font_tiny.render(label_txt, True, col)
                mx = (drone_px[0]+bx)//2;  my = (drone_px[1]+by)//2
                pad = 3
                pygame.draw.rect(surf, (10, 12, 22),
                    (mx-lsurf.get_width()//2-pad, my-lsurf.get_height()//2-pad,
                     lsurf.get_width()+pad*2, lsurf.get_height()+pad*2), border_radius=3)
                surf.blit(lsurf, (mx-lsurf.get_width()//2, my-lsurf.get_height()//2))

    # waypoints
    for wi, wp in enumerate(cfg.waypoints):
        cx, cy = to_px(wp[0], wp[1], cfg)
        is_cur = (wi == env.current_wp and not env.is_moving)
        is_tgt = (wi == env.target_wp  and env.is_moving)
        col    = WP_COL if is_cur else ((100, 180, 255) if is_tgt else WP_RING)
        pygame.draw.circle(surf, col,   (cx, cy), 12)
        pygame.draw.circle(surf, WHITE, (cx, cy), 12, 1)
        lbl = font_tiny.render(f"W{wi+1}", True, BG)
        surf.blit(lbl, (cx-lbl.get_width()//2, cy-lbl.get_height()//2))

    # base stations
    for bi, bs in enumerate(cfg.base_stations):
        cx, cy = to_px(bs.position[0], bs.position[1], cfg)
        d      = math.hypot(wp_pos[0]-bs.position[0], wp_pos[1]-bs.position[1])
        active = (bi == env.active_bs and d <= bs.radius and not env.is_moving)
        col    = BS_COL if active else BS_RING
        pygame.draw.circle(surf, col,   (cx, cy), 14)
        pygame.draw.circle(surf, WHITE, (cx, cy), 14, 2 if active else 1)
        surf.blit(font_tiny.render(f"BS{bi+1}", True, WHITE),
                  (cx-font_tiny.size(f"BS{bi+1}")[0]//2, cy-6))
        # show channel label below when actively uploading to this BS
        if active:
            ch_col = BEAM_COLS[env.active_ch]
            ch_lbl = font_tiny.render(f"Ch{env.active_ch+1}", True, ch_col)
            surf.blit(ch_lbl, (cx-ch_lbl.get_width()//2, cy+18))

    # sensors
    for si, sen in enumerate(cfg.sensors):
        ji   = sen.job_idx
        busy = (env.active_sensor == si)
        col  = SENSOR_DONE if env.job_collected[ji] else (SENSOR_BUSY if busy else SENSOR_COL)
        cx, cy = to_px(sen.position[0], sen.position[1], cfg)
        pygame.draw.circle(surf, col,   (cx, cy), 10)
        pygame.draw.circle(surf, WHITE, (cx, cy), 10, 1)
        surf.blit(font_tiny.render(f"S{si+1}", True, BG),
                  (cx-font_tiny.size(f"S{si+1}")[0]//2, cy-5))
        rem  = env.job_data_remaining[ji]
        dlbl = font_tiny.render(f"{rem:.2f}MB", True, col)
        surf.blit(dlbl, (cx-dlbl.get_width()//2, cy+13))
        # progress arc while downloading
        if busy:
            prog = min(1.0, env.collect_progress / max(0.01, cfg.jobs[ji].total_data))
            arc_rect = pygame.Rect(cx-13, cy-13, 26, 26)
            end_a = -math.pi/2 + prog * 2*math.pi
            if end_a > -math.pi/2:
                pygame.draw.arc(surf, CYAN, arc_rect, -math.pi/2, end_a, 2)

    # drone
    dx, dy = drone_px
    col    = DRONE_MOVE if env.is_moving else DRONE_COL
    glow   = pygame.Surface((40, 40), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*col, 50), (20, 20), 18)
    surf.blit(glow, (dx-20, dy-20))
    pygame.draw.circle(surf, col,   (dx, dy), 9)
    pygame.draw.circle(surf, WHITE, (dx, dy), 9, 2)
    surf.blit(font_tiny.render("D", True, WHITE),
              (dx-font_tiny.size("D")[0]//2, dy-5))

    pygame.draw.rect(surf, PANEL_EDGE, (CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H), 1)


#  main 

def main():
    pygame.init()
    pygame.display.set_caption("Drone Data Relay  Heuristic Demo")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    font      = pygame.font.SysFont("consolas", 16, bold=True)
    font_sm   = pygame.font.SysFont("consolas", 13, bold=False)
    font_tiny = pygame.font.SysFont("consolas", 11, bold=False)
    fonts     = (font, font_sm, font_tiny)

    cfg = EnvConfig()
    cfg.jobs = [
        JobConfig(total_data=5.0, deadline=4000, sensor_idx=0),
        JobConfig(total_data=8.0, deadline=5000, sensor_idx=1),
        JobConfig(total_data=4.0, deadline=3500, sensor_idx=2),
    ]
    #  Sensors along bottom edge  also serve as waypoints W1/W2/W3 
    cfg.sensors = [
        SensorConfig(np.array([20.0, 5.0]), radius=10.0, job_idx=0),
        SensorConfig(np.array([50.0, 5.0]), radius=10.0, job_idx=1),
        SensorConfig(np.array([80.0, 5.0]), radius=10.0, job_idx=2),
    ]
    # W1-W3 = sensor waypoints (bottom), W4-W5 = mid-field, W6-W8 = upload spots (near BSes)
    cfg.waypoints = np.array([
        [20.0,  5.0],   # W1  sensor S1
        [50.0,  5.0],   # W2  sensor S2
        [80.0,  5.0],   # W3  sensor S3
        [15.0, 50.0],   # W4  mid-left
        [85.0, 50.0],   # W5  mid-right
        [10.0, 76.0],   # W6  upload near BS1
        [50.0, 76.0],   # W7  upload near BS2
        [90.0, 76.0],   # W8  upload near BS3
    ])
    from .drone_env import BaseStationConfig
    # BSes at top, radius 20  rings just touch, no overlap
    cfg.base_stations = [
        BaseStationConfig(np.array([10.0, 95.0]), radius=20.0),
        BaseStationConfig(np.array([50.0, 95.0]), radius=20.0),
        BaseStationConfig(np.array([90.0, 95.0]), radius=20.0),
    ]
    cfg.max_steps = 6000
    cfg.alpha = 0.05   # upload rate /10 (total 100x slower than original)
    cfg.beta  = 0.015

    env    = DroneDataRelayEnv(config=cfg)
    obs, _ = env.reset(seed=0)
    agent  = HeuristicAgent(env)

    px       = to_px(cfg.waypoints[0][0], cfg.waypoints[0][1], cfg)
    drone_x  = float(px[0]); drone_y = float(px[1])

    paused  = False
    speed   = 1
    done    = False
    fps_val = 60.0

    while True:
        dt      = clock.tick(60) / 1000.0
        fps_val = 0.9*fps_val + 0.1/max(dt, 1e-6)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    obs, _ = env.reset(seed=int(time.time())%1000)
                    agent  = HeuristicAgent(env)
                    px     = to_px(cfg.waypoints[0][0], cfg.waypoints[0][1], cfg)
                    drone_x, drone_y = float(px[0]), float(px[1])
                    done   = False
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed+1, 30)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed-1, 1)

        if not paused and not done:
            for _ in range(speed):
                action = agent.act()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    done = True; break

        # smooth drone interpolation
        if env.is_moving:
            total = env._transit_time(env.current_wp, env.target_wp)
            frac  = 1.0 - env.transit_remaining/total if total > 0 else 1.0
            sx, sy = to_px(cfg.waypoints[env.current_wp][0], cfg.waypoints[env.current_wp][1], cfg)
            tx, ty = to_px(cfg.waypoints[env.target_wp][0],  cfg.waypoints[env.target_wp][1],  cfg)
            drone_x = sx + (tx-sx)*frac; drone_y = sy + (ty-sy)*frac
        else:
            tx, ty  = to_px(cfg.waypoints[env.current_wp][0], cfg.waypoints[env.current_wp][1], cfg)
            drone_x += (tx-drone_x)*0.25; drone_y += (ty-drone_y)*0.25

        drone_px = (int(drone_x), int(drone_y))

        screen.fill(BG)
        draw_canvas(screen, env, drone_px, fonts)
        draw_panel(screen, fonts, env, fps_val, speed)

        if done:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0,0,0,120))
            screen.blit(overlay, (0,0))
            msg1 = font.render(
                f"Episode ended  {int(env.job_done.sum())}/3 done, {int(env.job_failed.sum())} missed",
                True, WHITE)
            msg2 = font_sm.render("Press R to restart", True, MUTED)
            screen.blit(msg1, (WIN_W//2-msg1.get_width()//2, WIN_H//2-20))
            screen.blit(msg2, (WIN_W//2-msg2.get_width()//2, WIN_H//2+14))

        pygame.display.flip()


if __name__ == "__main__":
    main()