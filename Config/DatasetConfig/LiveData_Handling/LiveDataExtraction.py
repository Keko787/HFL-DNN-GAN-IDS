import pandas as pd
from scapy.all import *
import numpy as np


def extract_features(pcap_file):
    packets = rdpcap(pcap_file)
    flows = {}

    #
    # Group packets into flows
    #
    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            # -- Create flow key (bi-directional)
            # Sorting TCP / UDP
            if TCP in pkt:
                proto = 'TCP'
                proto_num = 6
                flags = pkt[TCP].flags
            else:
                proto = 'UDP'
                proto_num = 17
                flags = 0

            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst

            if proto == 'TCP':
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            else:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            # Create bi-directional flow key
            if f"{src_ip}:{src_port}-{dst_ip}:{dst_port}" in flows:
                flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                direction = "forward"
            elif f"{dst_ip}:{dst_port}-{src_ip}:{src_port}" in flows:
                flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}"
                direction = "backward"
            else:
                flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                flows[flow_key] = {
                    'packets': [],
                    'start_time': float(pkt.time),
                    'proto': proto,
                    'proto_num': proto_num
                }
                direction = "forward"

            # -- Add packet to flow
            flows[flow_key]['packets'].append({
                'time': float(pkt.time),
                'size': len(pkt),
                'flags': flags,
                'direction': direction
            })

    #
    # Extract features from flows
    #
    features = []
    for flow_key, flow_data in flows.items():
        packets = flow_data['packets']
        if len(packets) < 2:
            continue

        # Sort packets by time
        packets.sort(key=lambda x: x['time'])

        # -- Calculate flow duration
        duration = packets[-1]['time'] - packets[0]['time']
        if duration == 0:
            duration = 0.001  # Avoid division by zero

        # -- Calculate packet rates
        rate = len(packets) / duration

        # -- Calculate IAT (Inter-Arrival Time)
        iats = []
        for i in range(1, len(packets)):
            iats.append(packets[i]['time'] - packets[i - 1]['time'])
        iat_mean = np.mean(iats) if iats else 0

        # -- Count flags (for TCP)
        fin_count = 0
        syn_count = 0
        psh_count = 0
        if flow_data['proto'] == 'TCP':
            for p in packets:
                if p['flags'] & 0x01:  # FIN
                    fin_count += 1
                if p['flags'] & 0x02:  # SYN
                    syn_count += 1
                if p['flags'] & 0x08:  # PSH
                    psh_count += 1

        # -- Identify application protocols
        http = 0
        https = 0
        dns = 0
        telnet = 0
        smtp = 0
        ssh = 0
        irc = 0

        src_ip, src_port = flow_key.split('-')[0].split(':')
        dst_port = int(flow_key.split('-')[1].split(':')[1])

        if dst_port == 80 or src_port == '80':
            http = 1
        elif dst_port == 443 or src_port == '443':
            https = 1
        elif dst_port == 53 or src_port == '53':
            dns = 1
        elif dst_port == 23 or src_port == '23':
            telnet = 1
        elif dst_port == 25 or src_port == '25':
            smtp = 1
        elif dst_port == 22 or src_port == '22':
            ssh = 1
        elif dst_port == 6667 or src_port == '6667':
            irc = 1

        # -- Calculate total size and header length
        total_size = sum(p['size'] for p in packets)
        header_length = sum(p['size'] - (p['size'] - 40) for p in packets)  # Approximation

        features.append({
            'Protocol Type': flow_data['proto_num'],
            'Duration': duration,
            'Rate': rate,
            'Drate': 0,  # Would need more complex calculation
            'fin_flag_number': 1 if fin_count > 0 else 0,
            'syn_flag_number': 1 if syn_count > 0 else 0,
            'psh_flag_number': 1 if psh_count > 0 else 0,
            'HTTP': http,
            'HTTPS': https,
            'DNS': dns,
            'Telnet': telnet,
            'SMTP': smtp,
            'SSH': ssh,
            'IRC': irc,
            'TCP': 1 if flow_data['proto'] == 'TCP' else 0,
            'UDP': 1 if flow_data['proto'] == 'UDP' else 0,
            'DHCP': 1 if dst_port == 67 or src_port == '67' else 0,
            'ARP': 0,  # ARP is not IP
            'ICMP': 0,  # Would check separately
            'IPv': 1,  # All these flows are IPv4
            'Header_Length': header_length,
            'IAT': iat_mean
        })

    return pd.DataFrame(features)


# Usage Example:
df = extract_features("../../../../datasets/LIVEDATA/PCAP/test3.pcapng")
print(df.head())

# Save the extracted features to a CSV file
df.to_csv("../../../datasets/LIVEDATA/CSV/extracted_features.csv", index=False)
print("Features saved to extracted_features.csv")
