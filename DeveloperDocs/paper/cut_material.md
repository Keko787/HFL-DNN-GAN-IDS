# Material cut to fit 10 pages

**What this is:** the IPDPS 2027 paper had to shrink from about 16 body pages to
10, and the call allows no appendices at submission. This file keeps what the
trim removed, as source for the reproducibility appendix required after
acceptance. Read it with `methods_audit.md`, whose section C lists the
implementation details the paper never carried.

## Methodology (full text before the trim, commit `b1e8b07`)

The condensed section in `methods.tex` keeps every rule and number the
evaluation depends on. These were cut or shortened:
- the paragraph relating the plan score, the flight reward and the merge weight;
- FQ's feature groups and training reward in full;
- the time-varying backhaul's formula and the adaptive controller's utility;
- the contact-outcome and device-training-time details;
- the learning task's data split and model hyperparameters;
- the age-cap pilot rule;
- the route-search modes in full;
- the code paths the evaluation does not use;
- FerrySim's full list of differences.

The full text follows, as LaTeX.

```latex
\section{\sysname{} Methodology}
\label{sec:method}

\sysname{} sustains federated learning across regions that a stationary network
cannot reach, by sending an autonomous mule (UAV) between edge devices and an
edge server. This section describes what the mule decides, when it decides it,
and which quantities each decision uses. We organize the description by
\emph{decision clock} rather than by layer, because the two decisions that
determine most of the behavior of the system each span the radio, scheduling,
and learning layers. We first fix the setting and the role of each layer, then
define the quantities the layers supply, the shared feasibility test, the two
decision clocks, and aggregation, and close with the complete mission round,
its cost, and the implementation.

% -----------------------------------------------------------------------------
\subsection{Setting, Layer Roles, and Decision Clocks}
\label{sec:method:roles}

\paragraph{Setting}
A cluster consists of one edge server, one or more mules, and a set of field
devices. Devices are static, and the server's device registry holds their
positions, which the planner uses. The server assigns each mule a disjoint slice
$\mathcal{M}$ of the registry, fixed for the trial. Devices train their local model offline between
visits, so a contact is exchange only: the mule pushes the current global model
and pulls the device's prepared update. A \emph{mission} has two passes. In
\textsc{Pass 1} the mule collects updates along a route, merges them on board,
and docks to upload the merged result. The server merges across mules and
produces the next global model. In \textsc{Pass 2} the mule delivers that model
to every device in its slice. Delivering a known model version in Pass~2 is what
gives each collected update an observable basis version, and therefore an age.
Each mission has a time budget
$T_{\mathrm{miss}}$ for Pass~1: the plan and every check in flight require the
predicted landing and upload to fall within $T_{\mathrm{miss}}$ of takeoff. The
prediction prices the mean channel, so a realized mission can overrun.
Pass~2 is not budgeted.

\paragraph{Several mules}
With $K>1$ mules, the server cuts the field into $K$ contiguous sectors by angle
around the dock, starting after the widest empty arc, with slice sizes that
differ by at most one device. Every mule starts at the shared dock, and the mules
fly concurrently. Each mule plans and flies its own slice, and the server folds
the uploads in simulated-time order. Queueing at
the dock is not modelled.

\paragraph{Learning task}
The devices train a binary intrusion detector on CICIoT2023~\cite{CiCIoT2023}.
Every flow is labelled benign or attack (the 33 attack labels form one class) and
is described by 21 features scaled to $[0,1]$. Each trial draws 20{,}000
training flows, half of them benign, and a disjoint test set of 6{,}000 flows,
4{,}000 of them benign, so labelling every test flow benign scores 0.667. The training flows are split evenly at random across the
$N$ devices, about 3{,}333 per device at $N=6$. The model is a multilayer
perceptron with hidden layers of 64, 32, 16, 8 and 4 units (ReLU, batch
normalization, dropout 0.4, $L_2$ weight $10^{-3}$) and a sigmoid output, trained
with Adam (learning rate $10^{-3}$) on binary cross-entropy. A device trains one
local epoch (batch 64) per model it receives: offline after each Pass-2 delivery,
or, at no simulated cost, inside a Pass-1 contact at which it holds no prepared
update (its first contact, or one after a missed delivery). After each round
closes, the server evaluates the global model on the test set, stamped at the
simulated time the closing upload completed, before Pass~2 delivers the model;
the accuracy at a 0.5 threshold is the quantity the accuracy target $\tau$
refers to. The model occupies 18.8~KB; to represent a larger model, every
transfer is priced as if it held $S=1$~MB.

\paragraph{What the layers contribute}
Table~\ref{tab:layer-roles} states the role of each layer in the decisions
below. The radio layer prices a link; it does not choose among devices. The
scheduling layer makes every route and band decision. The learning layer
supplies the value of a collected update and the cost of an uncollected one.
Hard gates (eligibility, the feasibility test defined below, and the age cap)
always run before any ranking
step, so a ranking rule, learned or not, only chooses among options that the
gates have admitted.

\begin{table}[t]
\centering
\caption{Role of each layer in the joint decisions.}
\label{tab:layer-roles}
\small
\begin{tabularx}{\columnwidth}{l l L}
\toprule
\textbf{Layer} & \textbf{Role} & \textbf{Supplies to the decisions} \\
\midrule
L1 RF link & prices & Range, rate, dwell, and SNR over time for each band class; backhaul carrier \\
L2 Scheduling & decides & Admission, band class $\bbar$, route, per-arrival (band, next stop), re-plan \\
L3 Federated learning & values & Merge weight, update age, coverage weight, cross-mule merge \\
\bottomrule
\end{tabularx}
\end{table}

\paragraph{Two decision clocks}
The mule decides on two clocks, shown in Fig.~\ref{fig:clocks}. On the
\emph{plan clock}, once per mission at the dock, it chooses a contact band class
$\bbar$ and an ordered route together, because the band class fixes the contact
range, the range fixes how devices cluster into stops, and the stops fix the
route. On the \emph{flight clock}, at every Pass-1 arrival, it chooses the band
for the current stop and the next stop together, using the signal it observes at
that moment. The separation matters because the channel varies within a single
sortie: interference has a period comparable to the duration of a leg, so the
band that is best at takeoff need not be best at a later stop. A third element,
the \emph{return path}, carries what the flight observes back to the plan. Rate
sets dwell, dwell advances the mission clock, the clock decides feasibility, and
infeasibility triggers a re-plan.

\begin{figure}[t]
\centering
\begin{tikzpicture}[
  box/.style={draw, rounded corners=2pt, minimum height=0.62cm, align=center,
              font=\scriptsize, inner sep=2.5pt},
  gate/.style={box, fill=gray!15},
  arr/.style={-{Latex[length=1.6mm]}, thick},
  node distance=0.32cm]
  % plan clock
  \node[font=\scriptsize\bfseries, anchor=west] (pl) at (0,1.55) {Plan clock (dock, once per mission)};
  \node[box] (d) at (0.9,0.95) {demand,\\weights};
  \node[box, right=of d] (bb) {band class\\$\bbar$};
  \node[box, right=of bb] (st) {stops at\\$R(\bbar)$};
  \node[box, right=of st] (rt) {route\\$\pi$};
  \node[gate, right=of rt] (g1) {gate\\(S3b)};
  \draw[arr] (d)--(bb); \draw[arr] (bb)--(st); \draw[arr] (st)--(rt); \draw[arr] (rt)--(g1);
  % flight clock
  \node[font=\scriptsize\bfseries, anchor=west] (fl) at (0,0.1) {Flight clock (each Pass-1 arrival)};
  \node[box] (ob) at (0.9,-0.5) {observe\\SNR per band};
  \node[box, right=of ob] (pr) {(band, next\\stop) pair};
  \node[gate, right=of pr] (g2) {mask\\(S3b)};
  \node[box, right=of g2] (sv) {serve,\\dwell};
  \node[box, right=of sv] (rp) {fits?\\else re-plan};
  \draw[arr] (ob)--(pr); \draw[arr] (pr)--(g2); \draw[arr] (g2)--(sv); \draw[arr] (sv)--(rp);
  \draw[arr, dashed] (rp.south) -- ++(0,-0.32) -| (ob.south);
  \draw[arr, dashed] (g1.south) -- ++(0,-0.25) -| (ob.north west);
\end{tikzpicture}
\caption{The two decision clocks. The flight row shows FQ, which chooses the band
and the next stop together on arrival; FX chooses the band on arrival and the
next stop after the departure check, and F keeps the plan's. The shaded boxes
are the same feasibility test, applied once to the committed plan and again in
flight.}
\label{fig:clocks}
\end{figure}

\paragraph{What is learned}
Every decision on both clocks is made by a rule or a search over a declared
score. Learning can enter in one place only: a learned pair score can occupy the
flight slot in place of the fixed rule. We
therefore describe the design as a set of coupled decisions, and treat learning
as one optional way to fill one of them.

% -----------------------------------------------------------------------------
\subsection{Quantities Supplied by the Layers}
\label{sec:method:models}

\subsubsection{L1: Contact Link, Clock, and Backhaul}
\label{sec:method:l1}

\paragraph{Band classes}
The contact link offers a set $\mathcal{B}$ of band classes that differ in
occupied bandwidth, not in carrier. The three candidate carriers in our testbed
differ by at most 1.4~dB of free-space loss, which is too small to support a
range trade, so all classes share one carrier. We use three LTE channel
bandwidths~\cite{3gpp36104}: \textsf{wide} (20~MHz, 100 resource blocks),
\textsf{medium} (5~MHz, 25), and \textsf{narrow} (1.4~MHz, 6). Every rate and
range below uses the \emph{occupied} bandwidth $B_b=N_{\mathrm{RB}}\times
180$~kHz (18, 4.5 and 1.08~MHz). With a common transmit power and a noise floor
that scales with $B_b$, a narrower class reaches farther at a lower rate.
Relative to a reference class with bandwidth $B_0$ and slant range $R_0^{3D}$,
\begin{equation}
R_b^{3D} = R_0^{3D}\left(\frac{B_0}{B_b}\right)^{1/n},
\qquad
R_b = \sqrt{\left(R_b^{3D}\right)^2 - h^2},
\label{eq:range}
\end{equation}
where $n$ is the path-loss exponent and $h$ the UAV altitude. We anchor the
\textsf{wide} class at a planar range of 60~m with $h=25$~m and $n=2.2$, which
gives planar ranges of 60, 120 and 232~m for the three classes. This range is an
edge-availability range: at distance $R_b$ the mean SNR exceeds the decoding
floor by $M_{\mathrm{sh}}=\Phi^{-1}(0.9)\,\sigma_{\mathrm{sh}}=5.1$~dB, so a
device at the edge of its class is reachable with probability 0.9 under
shadowing alone (about 0.82 once the interference below is added).

\paragraph{Rate and dwell}
The mean SNR at distance $d$ under class $b$ is
\begin{equation}
\bar\gamma_b(d) = \gamma_{\mathrm{floor}} + M_{\mathrm{sh}}
  + 10\,n\log_{10}\!\frac{R_b^{3D}}{d^{3D}},
\label{eq:meansnr}
\end{equation}
with $\gamma_{\mathrm{floor}}=-6.7$~dB (CQI~1). The achievable rate is the
smaller of a CQI-table spectral efficiency and the Shannon rate on the occupied
bandwidth, and is zero below the floor:
\begin{equation}
r_b(\gamma)=\begin{cases}
\min\bigl\{\kappa_b B_b\,\mathrm{SE}(\gamma),\\
\quad B_b\log_2(1+10^{\gamma/10})\bigr\}, & \gamma\ge\gamma_{\mathrm{floor}},\\
0, & \text{otherwise,}
\end{cases}
\label{eq:rate}
\end{equation}
where $\kappa_b$ is an implementation efficiency factor and $\mathrm{SE}(\cdot)$
the CQI efficiency table. Transferring $S_j$ bytes to or from device $j$
therefore takes
\begin{equation}
\tau_{j,b}^{\mathrm{dwell}}(\gamma)=\frac{8S_j}{r_b(\gamma)}.
\label{eq:dwell}
\end{equation}
A Pass-1 session pushes the global model and pulls the update, so it is priced at
$S_j=2S$; a Pass-2 delivery and the upload at the dock are each priced at $S$.
The dwell times of the devices served at one stop add, because they share one
channel. A device whose SNR is below the floor at arrival is not solicited and
costs no airtime. The channel acts on a contact only through these two effects,
whether a device is solicited and how long its session takes: each session's
rate is read once, at its start, the sessions at a stop run one after another,
and a session that has started always completes, priced at the floor rate if
the SNR has fallen below the floor. The messages themselves travel over a
lossless local TCP link that carries no radio effects.

\paragraph{Channel over time}
The realized SNR of device $j$ on class $b$ at mission time $t$ is
\begin{equation}
\gamma_{b,j}(t)=\bar\gamma_b(d_j)+X_j(t)+I_b(t),
\label{eq:realsnr}
\end{equation}
where $X_j(t)$ is correlated shadowing ($\sigma_{\mathrm{sh}}=4$~dB, correlation
time 7.4~s) and
\begin{equation}
I_b(t)=A\sin\bigl(2\pi(t/P_c+\phi_b)\bigr)+\sigma_I\,\xi_b(t)
\label{eq:interf}
\end{equation}
is a per-class interference term~\cite{3gpp36777}: a periodic swing with period
$P_c=60$~s and a class-specific phase $\phi_b\in\{0,\tfrac13,\tfrac23\}$ of a
period, plus noise $\xi_b(t)$. In the interference-limited (``jittery'') regime
used throughout the evaluation, $A=5$~dB and $\sigma_I=1.5$~dB. Both terms are
generated from keyed hashes, so two policies that ask for the same value at the
same time receive the same value: shadowing is keyed by (seed, device, time) and
shared across classes, and the interference noise by (seed, class, time) and
shared across devices. The planner prices only the mean $\bar\gamma_b$. Pricing
the realized phase at plan time would be an oracle, so realized missions run
longer than planned, by an amount that grows as the class narrows. The flight
clock observes $\gamma_{b,j}(t)$. The planner also uses the probability that a
served device falls below the floor,
\begin{equation}
p_{\mathrm{out}}(b,d)=\Phi\!\left(\frac{\gamma_{\mathrm{floor}}-\bar\gamma_b(d)}{\sigma_{\mathrm{eff}}}\right),
\quad
\sigma_{\mathrm{eff}}^2=\sigma_{\mathrm{sh}}^2+\sigma_I^2+\tfrac{A^2}{2},
\label{eq:pout}
\end{equation}
a moment-matched approximation of the shadowing-plus-interference outage.

\paragraph{Mission clock and energy}
Mission time is a simulated clock charged with the physics of each step:
transit at cruise speed (5~m/s), dwell from Eq.~\eqref{eq:dwell}, a listen
window when an expected reply is missing, the return leg, the upload, and a
fixed dock turnaround of 30~s. Energy is a function of the same ledger,
\begin{equation}
E = P_{\mathrm{move}}\,(t_{\mathrm{transit}}+t_{\mathrm{return}})
  + P_{\mathrm{hover}}\,(t_{\mathrm{dwell}}+t_{\mathrm{listen}}),
\label{eq:energy}
\end{equation}
with $P_{\mathrm{move}}=143.6$~W and $P_{\mathrm{hover}}=168.5$~W from the
rotary-wing model of Zeng et al.~\cite{zeng2019energy}. This energy is
simulated, not measured, and no battery capacity is enforced in the evaluated
configuration. Flight is planar at the fixed altitude $h$, with no takeoff,
landing or climb time or energy, and the energy counts propulsion only.

\paragraph{Backhaul}
The upload from mule to server rides one of three backhaul carriers, each with a
gain $g(c)\sim U(0,3)$~dB, and the evaluation uses two models of it. In the
main configuration the mule holds the carrier with the largest gain; the upload
is timed at that carrier's mean SNR and is lost with a fixed probability (2\% in
the jittery regime), drawn per upload from a seeded stream. A lost upload is not
retried: its updates are discarded, the server returns the unchanged model,
which Pass~2 delivers, and the planner, which records the devices as merged
before the upload, still counts them as served. Study~5.14's backhaul test and
every cell of Study~5.15 use a time-varying
backhaul instead: carrier $c$'s SNR at the upload time $t$ is
$6+g(c)+5\sin\bigl(2\pi(t/P_{\mathrm{bh}}+\phi_c)\bigr)+1.5\,\xi_c(t)$~dB, with
$P_{\mathrm{bh}}=4\Tnom$, and an upload is lost with probability
$1/\bigl(1+e^{(\gamma_c(t)-3)/2}\bigr)$. On the fixed carrier this loses about
a fifth of the uploads: F closes 79\% of its rounds there, against 97\% in the
main configuration. Only on this backhaul does the radio layer's adaptive
controller act (F+L1): before each upload it picks the carrier that maximizes
$U(c,t)=R\bigl(\gamma_c(t)\bigr)-\lambda\,\mathbb{1}[c\neq c_{\mathrm{prev}}]$,
with $R(\gamma)=\log_2(1+10^{\gamma/10})$ above 0~dB and 0 below, and a
switching cost $\lambda=0.5$. It observes the realized SNR that decides the
loss, so it approaches a per-upload oracle. The backhaul never enters a
scheduling decision.

\subsubsection{L2/L3: Demand, Deadlines, Ages, and the Age Cap}
\label{sec:method:demand}

\paragraph{Contact outcomes}
Each device served in Pass~1 ends its contact in one of three outcomes. It is
\emph{clean} when the device returns an update that matches the round, size and
checksum the mule expects, and \emph{partial} when it answers but refuses the
session or returns an update that fails those checks. Every other contact is a
\emph{timeout}: the device lies beyond the band's reach or below the decoding
floor at arrival, stays silent, fails the push, sends no reply, or has no update
ready. Whether a reachable device replies is drawn each mission from its
reliability $\rho_j$, drawn once per device and trial from $U(0.15,1)$. A device
whose reply is lost still receives the push, at its airtime, and adopts the
pushed model; a stop with any missing reply costs one 1~s listen window on the
simulated clock. A session also has a wall-clock timeout (36~s at $N=6$, set by
a pilot as twice the 95th-percentile fit time), and a session that outlasts it
is recorded as a timeout. Host load could therefore change an outcome; at 0.75,
1 and 1.5 times the timeout, a sensitivity run gave identical results.

\paragraph{Device training time}
By default local training costs no simulated time, so an update is always ready
when the mule arrives. To study compute heterogeneity, a device can instead
take
$T_j=\tilde{T}\,e^{\sigma z_j}$ to train, with median $\tilde{T}$, $\sigma=0.5$,
and $z_j$ a keyed standard normal draw; a fixed share of the devices can be made
stragglers whose time is multiplied by a factor. A device's first fit starts at
the mule's first takeoff, and each model it receives restarts the fit. A device
whose update is not ready when the mule arrives answers but has nothing to send:
the contact is a timeout that costs no airtime and widens the device's window by
$\beta_{\mathrm{partial}}$ (Eq.~\eqref{eq:phi}). The plan does not price
training time.

\paragraph{Per-device deadline}
Each device $j$ carries a fulfilment window $\Phi(j)$ and a deadline
\begin{equation}
\mathrm{Deadline}(j)=t+\Phi(j)-\iota(j),
\label{eq:deadline}
\end{equation}
where $t$ is the planning time and $\iota(j)$ the time since the device's last
clean contact. The window starts at $\Phi_0=60\,\mathrm{s}\times\sigma$, where
the time scale $\sigma=\Tnom/10\,\mathrm{s}$ converts the unit in which the law
was first set to the duration of a simulated mission. $\Tnom$ is the median,
over 20 reference layouts of the cell, of the duration of one nominal two-pass
mission on the wide band with no budget and no failures (203~s at $N=6$ with one
mule). After each contact the window is multiplied by a factor that depends on
the outcome and clamped,
\begin{equation}
\Phi(j)\leftarrow\min\{\Phi_{\max},\max\{\Phi_{\min},\,\beta_{o}\,\Phi(j)\}\},
\label{eq:phi}
\end{equation}
with $\beta_{\mathrm{clean}}=0.8$, $\beta_{\mathrm{partial}}=1.25$,
$\beta_{\mathrm{timeout}}=1.5$, $\Phi_{\min}=5\sigma$~s, and
$\Phi_{\max}=300\sigma$~s. A timeout after the device answered the solicitation
(a failed push, a lost reply, or no update ready) widens the window by
$\beta_{\mathrm{partial}}$, as a partial does; $\beta_{\mathrm{timeout}}$
applies to a device that stayed silent, was out of reach, or was left out. A
clean contact re-anchors $\iota(j)$; a partial or
timed-out contact widens the window but leaves the anchor stale, so a device
that keeps failing grows steadily more overdue. A device that the plan leaves
out, or that the mule drops in flight, is recorded as a synthetic timeout. The
server can in principle override a deadline at the dock, but the evaluated
configuration issues no overrides, so the window adapts only in flight.

\paragraph{Ages}
Two ages are used, in different units and for opposite purposes. The \emph{plan
age} $a_j=m-U_j$ counts the missions of the device's own mule since the mission
$U_j$ whose merge last used an update from $j$ ($U_j=0$ if none). A high plan age
strengthens a device's claim to be served. The \emph{merge age} of an update is
$a_i=v-b_i$, the version $v$ of the global model the mule carries minus the
version $b_i$ of the model the device trained from; it is measured in cluster
rounds and discounts the update's weight in the merge. The
device echoes the basis version with its update, so the mule can compute the age
without holding old models.

\paragraph{Priority is separate from the deadline}
A missed contact widens $\Phi(j)$, which relaxes the device's \emph{time window}.
It does not lower the device's \emph{priority}. Priority enters through the
coverage weight
\begin{equation}
w_j=\max(a_j,1)\,(1+\mu_j),
\label{eq:covweight}
\end{equation}
where $\mu_j$ is the device's miss streak, and through the age cap below. Under
a plan, every device left out is recorded as a miss, so $\mu_j=a_j-1$ and the
weight grows roughly as $a_j^2$.

\paragraph{Age cap}
A device is \emph{capped} when $a_j\ge S-L$, with cap $S$ and lookahead $L=0$. We
set $S$ with a pilot rule. For each of 30 sampled layouts, $S^\ast$ is the fewest
missions that together serve every servable device (exact up to six devices, a
greedy upper bound above). $S$ is the smallest value, at least two, that covers
90\% of the layouts at both evaluation budgets. One mission suffices at the knee
budget; the stress budget sets $S=2$ at every $N$. Capping has four effects. A
stop's own deadline is the earliest among its uncapped members, and a stop whose
members are all capped is exempt from its deadline clause but not from the
budget. Any stop with a capped member is flown first when a route is
trimmed. A capped device that its cluster stop cannot serve alone within the
budget is offered a one-device \emph{hover stop} on the segment from the dock to
the device. And the plan key, defined below, compares the ages
of the capped devices that a plan leaves out before it compares anything else.

% -----------------------------------------------------------------------------
\subsection{The Feasibility Test and the Mission Clock}
\label{sec:method:gate}

One predicate decides whether a stop may be served next, and it is the only
admission rule in the plan-mode system. It is applied to the committed plan,
inside the plan search, at every departure, inside the re-plan, and to the
in-flight choices of the flight clock. For a stop $s$ with member set $\mathcal{M}_s$, reached
from a state (pose, clock $c$, energy) on band $b$,
\begin{align}
t_{\mathrm{arr}} &= c+\|\mathrm{pose}-s\|/v,\\
t_{\mathrm{fin}} &= t_{\mathrm{arr}}+\!\!\sum_{j\in\mathcal{M}_s}\!\tau^{\mathrm{dwell}}_{j,b}(\bar\gamma_b(d_j)),\\
t_{\mathrm{home}} &= t_{\mathrm{fin}}+t_{\mathrm{return}}+t_{\mathrm{upload}},
\end{align}
and the stop is admitted only if every clause holds, with the first failure
reported as the reason:
\begin{enumerate}
\item \emph{own deadline:} $t_{\mathrm{fin}}\le\min_{j}\mathrm{Deadline}(j)$ over the stop's uncapped members, waived for a stop with none;
\item \emph{budget:} $t_{\mathrm{home}}\le T_{\mathrm{end}}$, where $T_{\mathrm{end}}$ is takeoff plus $T_{\mathrm{miss}}$;
\item \emph{energy:} the energy of Eq.~\eqref{eq:energy} stays within capacity, if a capacity is set.
\end{enumerate}
(The predicate also has a route-level clause that bounds the landing by the
earliest deadline among updates already on board; it is not used in the
evaluated configuration.) Every clause is monotone in the member set: dwell is a
sum of non-negative terms, and arrival, return and upload do not depend on the
members. This allows a stop that fails whole to be reduced to the members that
fit in one pass. The test prices the mean SNR, so it can admit a stop that the
realized channel then lengthens; the departure check on the flight clock
catches this after the stop. Nothing re-checks after the last Pass-1 stop, so a
realized mission can overrun the budget.

% -----------------------------------------------------------------------------
\subsection{Plan Clock: Reach and Route}
\label{sec:method:plan}

At the dock the mule commits a plan $\mathcal{P}=(\bbar,\pi)$, where $\bbar$ is
the band class flown in both passes and $\pi$ is an ordered list of stops, each
reduced to the members that fit. The plan is built in five steps.

\paragraph{Demand}
The demand is every device in the mule's slice, each with its deadline, plan
age, miss streak, and weight from Eq.~\eqref{eq:covweight}. (The scheduler can
also admit devices through readiness beacons and server overrides; neither
occurs in the evaluated configuration.)

\paragraph{Reach is a decision}
For each class $b$ the demanded devices are clustered at the class's planar
range $R_b$. A greedy anchor sweep places a stop at the centroid of a cluster if
every member lies within $R_b$ of it, and otherwise at the anchor's position;
the stop's deadline is the tightest of its members'. A wide class serves fewer
devices per stop at a higher rate, whereas a narrow class can reach the field
from one stop and dwells longer there. No fixed rule chooses between them, so
the planner prices every class and commits the best.

\paragraph{Route search}
For each class, candidate routes are generated and passed through the
feasibility test, with each stop checked from
the state the previous stop leaves. The search depends on the problem size:
\begin{itemize}
\item \textsf{exact}, for at most six demanded devices: every ordered sequence of stops, each reduced to every non-empty member subset, pruning any failing prefix and any superset of a failing member set. It is optimal over the member subsets that the score prices.
\item \textsf{stop-subsets}, for more devices but at most six stops in the class: ordered subsets of stops, each kept whole if it fits and otherwise reduced greedily to the members that fit.
\item \textsf{local}, for more stops: a 2-OPT tour, then a member trim, then first-improvement moves (drop a stop, insert a stop, reverse a segment, drop a member), bounded by 50 passes and 2000 evaluations per class across all moves, so that a repeated trial plans the same.
\end{itemize}
Only the \textsf{exact} search is optimal. The other two are heuristics. The
empty plan is a candidate and is chosen only when no plan that serves any device
is admitted.

\paragraph{Plan score}
Candidates are scored by
\begin{equation}
\begin{split}
V(\bbar,\pi\mid\text{demand})
={}&-\Bigl[c_1\Bigl(\tfrac{\Delta}{\Tnom}\Bigr)^{2}+c_2U+c_3L\Bigr]\\
&-c_4\frac{E}{P_{\mathrm{hover}}\Tnom},
\end{split}
\label{eq:planscore}
\end{equation}
where $\Delta$ is the duration of the whole mission on $\bbar$ (Pass~1, return,
upload, turnaround, and Pass~2, which also flies $\bbar$), and
\begin{equation}
U=1-\frac{\sum_{j\in\text{served}}w_j}{\sum_{j\in\text{demand}}w_j},
\qquad
L=\frac{\sum_{j\in\text{served}}w_j\,p_{\mathrm{out}}(\bbar,d_j)}{\sum_{j\in\text{demand}}w_j}
\end{equation}
are the weighted coverage shortfall and the expected link loss, and $E$ is the
energy of Eq.~\eqref{eq:energy} over both passes. The constants are
$c_1=1$, $c_2=\kappa N_{\mathrm{demand}}$ with $\kappa=1$, $c_3=c_2$, and
$c_4=0.1$. The form follows the route-dependent bound of FedEx~\cite{bian2025indirect},
but the score is declared, not derived: the constants are set by hand, and the
convex term in $\Delta$ is a surrogate for staleness, not a bound for this
system, because a \sysname{} device trains once per visit.

\paragraph{Plan key}
The mule does not commit the plan with the highest $V$. Every plan that serves
any device pays for a whole Pass~2, whereas the empty plan pays for none, so $V$
alone can prefer to fly empty over serving a device that fits. The mule instead
commits the plan with the smallest key
\begin{equation}
\bigl(\kappa_{\mathrm{cap}},\;-\textstyle\sum_{\text{served}}w_j/\sum_{\text{demand}}w_j,\;-V,\;\text{class index}\bigr),
\label{eq:plankey}
\end{equation}
compared lexicographically, where $\kappa_{\mathrm{cap}}$ is the vector of ages
of capped devices that the plan leaves out, largest first. The key is a total
order, so the choice never depends on the order in which candidates were
enumerated. Under this key, time and energy break ties among plans that serve
the same weight.

\paragraph{Commit}
Before commit, a guard fold re-checks that the chosen route passes the
feasibility test as it will be flown; a failure raises an error and commits
nothing. Every demanded device that the plan leaves out is labelled with the
reason and recorded as a miss. The plan clock is a search, not a learned
decision: no plan-time value is learned.

% -----------------------------------------------------------------------------
\subsection{Flight Clock: Band, Next Stop, and Re-Planning}
\label{sec:method:flight}

\paragraph{The flight slot}
In Pass~1, after takeoff, the mule makes two choices per stop on the flight
clock: the band on which to serve the stop it has reached, chosen from the
realized SNR of every class on arrival, and the next stop of the remaining
queue. The slot does not act at takeoff, because nothing has been observed, or
in Pass~2, which flies $\bbar$ in queue order. \sysname{} with an empty slot,
which flies the plan's band and order unchanged, is F, the reference arm. The
evaluation fills the slot in two ways, which make the next-stop choice at
different moments.

\paragraph{Fixed filling (FX)}
The fixed filling is a cross-layer rule. On arrival it selects the fastest class
that still reaches every device $\bbar$ reaches at this stop, priced at the
observed SNR. At the next departure, after the stop is served and the departure
check has run, it selects the nearest remaining stop whose move to the front of
the queue still passes the feasibility test from the realized clock, or else the
plan's next stop. We call \sysname{} with this filling FX.

\paragraph{Learned filling (FQ)}
The learned filling makes both choices on arrival, as a pair $(b,s)$ ranked by a
learned score. A pair is admitted only if all three conditions hold:
\begin{enumerate}
\item the band reaches every device that $\bbar$ reaches at this stop, at the measured signal;
\item the next stop belongs to the plan, so the slot reorders but never invents stops;
\item the whole remaining flight still fits after serving here on $b$ at the observed dwell and flying to $s$, evaluated by the feasibility test on $\bbar$'s model.
\end{enumerate}
This bounds the pair set by $|\mathcal{B}|\times|\text{queue}|$, at most 18 pairs
for three classes and six stops. If no pair is admitted, the mule serves the
stop on FX's band and continues to the plan's next stop without reordering, and
records that the mask was empty. The mask matters most at the end of a sortie:
at six devices, FX's own pair would overrun the budget at about one last-stop
arrival in four. The score is a masked pointer double DQN (two hidden layers of
64 units) over a 36-feature row per admitted pair, which includes the SNR now,
the dwell, the next stop's mean SNR per class, deadline slack, plan ages, the
clock and energy left, and the arrival's phase in the interference cycle. It is
trained in FerrySim (described below) on a per-decision reward, the
merge weight collected at the stop less a time cost, with the weighted shortfall
of the devices the plan committed to charged at the end of each sortie that
flew a stop ($c_t=0.1$, $c_{\mathrm{cov}}=1$, hand-set); in training, a targeted
device is credited at its reliability $\rho_j$ rather than the realized draw. We
call \sysname{} with this filling FQ. FQ thus picks its next stop before
serving, from the priced dwell, and FX after serving, from the realized clock.

\paragraph{Departure check and re-plan}
After serving a stop, the mule folds the remaining queue through the feasibility
test at the clock it now has. If the queue fits, the mule proceeds to the stop
selected by the slot. If it does not, the mule re-plans the remainder: stops with
a capped member are kept first, then the plan's own order is kept and the stops
and members that cannot be served in that order are trimmed away. Moving the
stops with a capped member to the front is the re-plan's only reordering; any
other belongs to the flight slot.
Every dropped stop is final for the mission and is recorded as a synthetic
timeout, so its window widens.

% -----------------------------------------------------------------------------
\subsection{Aggregation: Mule, Cluster, and Delivery}
\label{sec:method:merge}

\paragraph{Merge on the mule}
Devices send the update $\Delta\theta_i=\theta_i^{\mathrm{after}}-\theta_i^{\mathrm{basis}}$
together with its basis version. At the end of Pass~1 the mule merges the updates
with age-dependent weights,
\begin{equation}
w_i=n_i\,v_i\,s(a_i)\ \text{ for } a_i\le a_{\max,j},\qquad w_i=0\ \text{ otherwise},
\label{eq:mergeweight}
\end{equation}
\begin{equation}
M=\sum_{i\,\text{admitted}}n_iv_i,\qquad
\Delta_m=\frac{1}{M}\sum_{i\,\text{admitted}}w_i\,\Delta\theta_i,
\label{eq:mulemerge}
\end{equation}
where $n_i$ is the number of local examples, $v_i$ a value proxy (uniform in the
evaluation), and $a_i$ the merge age. The staleness factor is the hinge form of
FedAsync~\cite{xie2019asynchronous}, $s(a)=1$ for $a\le b_h$ and
$s(a)=\bigl(a_h(a-b_h)+1\bigr)^{-1}$ otherwise, with $a_h=1$ and $b_h=0$. The
cutoff is derived from the device's deadline window, $a_{\max,j}=\lfloor
\Phi(j)/T\rfloor$ with $T=\Tnom$ the merge period, so the same window that
admits a device (Eq.~\eqref{eq:deadline}) also bounds how long its update
counts. The cutoff is read once after planning, so the mission's own clean
contacts cannot tighten it mid-mission. The normalizer $M$ does not contain the
staleness factors: dividing by $\sum w_i$ would cancel a common factor and let a
uniformly stale mission move the model by its full mean, whereas dividing by $M$
makes it move the model by $s(a)$ times its mean. With every basis current and
$v_i=1$, Eq.~\eqref{eq:mulemerge} reduces to the plain example-weighted mean.

\paragraph{Merge across mules}
At the server, the partial $\Delta_m$ from each mule carries its own staleness
$s_m=s(V-v_m)$, where $V$ is the server's current version and $v_m$ the version
that mule carried. The global model is updated as
\begin{equation}
\theta\leftarrow\theta+\eta\,\frac{\sum_{m\,\text{live}}M_m\,s_m\,\Delta_m}{\sum_{m\,\text{live}}M_m},
\label{eq:clustermerge}
\end{equation}
with server rate $\eta=1$ and no server optimizer. A partial is live if its
staleness factor is positive and it is non-empty. If none is live, the server
takes no step, keeps the round open, and still returns the waiting mules' next
bundle. In the evaluated configuration the server merges as soon as one partial
arrives (minimum participation of one), so each upload closes its own round and
no round waits on an unreachable device. A mule flies Pass~1 with the model it
received at its previous dock. With one mule that model is current; with
several, the other mules' merges in between make a partial arrive stale, and
the server applies no cutoff: at $N=6$ with three mules, 160 of 198 merges
applied a partial one to three versions old, discounted to between a half and a
quarter of its weight.

\paragraph{Delivery}
After the merge, if Pass~1 collected any update, the mule flies Pass~2: it
delivers the new global model to every stop of its slice (the contact clusters
at $R(\bbar)$), nearest first, on band $\bbar$, with no scheduler decision; it
skips only a device that is below the decoding floor on arrival, which is
rare. Each device stores the model together with its version and resumes
offline training against it.

\paragraph{Relation between the plan score, the reward, and the merge weight}
The plan score, the flight reward and the merge weight are built from the same
ingredients but are not one quantity. The merge weight falls with merge age,
whereas the coverage weight in $V$ rises with plan age, and the two ages use
different units. The reward uses the merge weight for its gain and the plan's
coverage weights for the shortfall. In the evaluated configurations most
collected updates have merge age zero on the mule (the server-side staleness of
several mules, above, is separate): all of them at $N=6$, while at $N=24$
under the stress budget 8 of 80 mule merges held a stale update, from a device
that missed a Pass-2 delivery. Where an update is stale, the cutoff and the
hinge apply.

% -----------------------------------------------------------------------------
\subsection{Mission Round, Cost, and Implementation}
\label{sec:method:alg}

\begin{algorithm}[t]
\caption{One \sysname{} mission round}
\label{alg:mission}
\begin{algorithmic}[1]
\Require Slice $\mathcal{M}$, model $\theta$ with version $v$, budget $T_{\mathrm{miss}}$, cap $S$
\Statex \textbf{[Dock: plan clock]}
\State Take slice $\mathcal{M}$ and the model $\theta$ (version $v$) received at the previous dock; compute demand, weights $w_j$, capped set
\For{each band class $b\in\mathcal{B}$}
  \State $\textit{stops}_b\gets$ cluster demand at $R_b$ and add hover stops for capped devices
  \State $\mathcal{P}_b\gets$ \textsc{Search}$(\textit{stops}_b)$ under the feasibility test
\EndFor
\State $(\bbar,\pi)\gets$ plan with the smallest key (Eq.~\ref{eq:plankey}); guard fold; commit; record left-out devices as misses
\Statex \textbf{[Pass 1: flight clock]}
\For{each stop $s_k$ in the queue}
  \State Fly to $s_k$; observe $\gamma_{b,j}(t)$ for all classes
  \State Band $b_k$: $\bbar$ (F); fastest covering class (FX); with $s_{k+1}$, best admitted pair (FQ)
  \State Serve members on $b_k$; update $\Phi(j)$ and $\iota(j)$ from each outcome
  \If{remainder does not pass the feasibility test}
    \State Re-plan by trimming; record dropped stops as timeouts
  \EndIf
  \State Next stop: plan order (F); nearest that keeps the remainder feasible (FX); the pair's (FQ)
\EndFor
\State Merge on the mule with cutoff (Eq.~\ref{eq:mulemerge}); fly to the dock; upload
\Statex \textbf{[Dock: server]}
\State Server merges across mules (Eq.~\ref{eq:clustermerge}) to obtain $\theta'$ with version $v+1$
\Statex \textbf{[Pass 2, if Pass 1 collected any update]}
\For{each stop in nearest-first order from the dock}
  \State Deliver $\theta'$ on $\bbar$; devices resume offline training
\EndFor
\end{algorithmic}
\end{algorithm}

\paragraph{Cost}
Local training stays on the devices. On the plan clock, the \textsf{exact} search
evaluates at most $1957$ ordered stop sequences per class for six one-device
stops, and the \textsf{local} search is bounded by 2000 evaluations and 50 passes
per class, so planning cost does not depend on wall time. On the flight clock,
each arrival evaluates at most $|\mathcal{B}|\times|\text{queue}|$ pairs, each
with one fold of the feasibility test. The merge on the mule is linear in the
number of collected updates, and the merge at the server is linear in the number
of mules.

\paragraph{Implementation and scope}
The prototype is a Python implementation in which the server, each mule, and
each device run as separate processes that exchange messages over TCP, and
devices train the real model. Mission time and energy come from the simulated
clock above, and the
radio quantities are simulated from the models above; none is a measured value
from the physical testbed. Updates carry integrity checks (round, size and a
SHA-256 checksum) but no authentication; device authentication,
poisoning-robust aggregation and privacy mechanisms are out of scope. The code
base also holds parts that the evaluated configuration does not use: the
device-readiness thresholds and beacon-driven admission of earlier versions of
the scheduler, a mission-level adaptation of the deadline windows, an earlier
learned tie-breaking selector, and a learned channel selector in the radio
layer.

\paragraph{FerrySim}
The learned per-stop score, the learned baseline E3, and the largest problems
($N=48$ and $96$) run in FerrySim, an in-process simulator built from the same
code. It runs the same driver; the same server, mule and device services; and
the same planner, flight slot, feasibility test, channel and mission clock, but
connects them by synchronous in-process links instead of TCP. It omits model
training: each device returns the global model plus small Gaussian noise, so an
episode (one four-mission trial) measures scheduling, not learning. An episode
is scored by FQ's training reward (E3's by the share of
devices collected), and training, validation and held-out episodes come from
separate seed streams. FerrySim also differs from the full system in these
ways: it flies one mule; its cells use an additive deadline law, no merge
cutoff and a 3~s session timeout; and its scale cells ($N=24$, 48 and 96) grow
the field with $N$ at $N=6$'s density (half-widths of 200, 283 and 400~m), with
budgets from their own pilot, so FerrySim's $N=24$ is not the full system's.
```
