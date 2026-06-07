import argparse
import time
from collections import Counter, defaultdict
from scapy.all import sniff, IP, TCP, UDP, ICMP, DNS, DNSQR, Raw, get_if_list

class TrafficAnalyzer:
    def __init__(self):
        self.start_time = time.time()
        self.total_packets = 0
        self.total_bytes = 0
        self.protocol_counter = Counter()
        self.src_ip_counter = Counter()
        self.dst_ip_counter = Counter()
        self.port_counter = Counter()
        self.dns_queries = Counter()
        self.http_hosts = Counter()
        self.http_paths = Counter()
        self.conversations = Counter()

    def analyze_packet(self, packet):
        self.total_packets += 1
        self.total_bytes += len(packet)

        if IP not in packet:
            self.protocol_counter["NON_IP"] += 1
            return

        ip = packet[IP]
        src_ip = ip.src
        dst_ip = ip.dst

        self.src_ip_counter[src_ip] += 1
        self.dst_ip_counter[dst_ip] += 1
        self.conversations[(src_ip, dst_ip)] += 1

        protocol_name = "OTHER"

        if TCP in packet:
            tcp = packet[TCP]
            sport = tcp.sport
            dport = tcp.dport

            protocol_name = "TCP"
            self.port_counter[dport] += 1

            if sport == 80 or dport == 80:
                protocol_name = "HTTP"
                self.extract_http(packet)

            elif sport == 443 or dport == 443:
                protocol_name = "TLS/HTTPS"

        elif UDP in packet:
            udp = packet[UDP]
            sport = udp.sport
            dport = udp.dport

            protocol_name = "UDP"
            self.port_counter[dport] += 1

            if DNS in packet:
                protocol_name = "DNS"
                self.extract_dns(packet)

        elif ICMP in packet:
            protocol_name = "ICMP"

        self.protocol_counter[protocol_name] += 1

        self.print_packet(packet, protocol_name)

    def extract_dns(self, packet):
        if DNSQR in packet:
            query = packet[DNSQR].qname.decode(errors="ignore").rstrip(".")
            if query:
                self.dns_queries[query] += 1

    def extract_http(self, packet):
        if Raw not in packet:
            return

        payload = packet[Raw].load

        try:
            text = payload.decode("utf-8", errors="ignore")
        except Exception:
            return

        if not (
            text.startswith("GET ")
            or text.startswith("POST ")
            or text.startswith("PUT ")
            or text.startswith("DELETE ")
            or text.startswith("HEAD ")
            or text.startswith("OPTIONS ")
        ):
            return

        lines = text.split("\r\n")
        request_line = lines[0]

        parts = request_line.split()
        if len(parts) >= 2:
            path = parts[1]
            self.http_paths[path] += 1

        for line in lines:
            if line.lower().startswith("host:"):
                host = line.split(":", 1)[1].strip()
                self.http_hosts[host] += 1
                break

    def print_packet(self, packet, protocol_name):
        if IP not in packet:
            return

        ip = packet[IP]
        src = ip.src
        dst = ip.dst
        length = len(packet)

        extra = ""

        if TCP in packet:
            tcp = packet[TCP]
            extra = f"{tcp.sport} -> {tcp.dport} flags={tcp.flags}"

        elif UDP in packet:
            udp = packet[UDP]
            extra = f"{udp.sport} -> {udp.dport}"

        elif ICMP in packet:
            icmp = packet[ICMP]
            extra = f"type={icmp.type} code={icmp.code}"

        print(f"[{protocol_name:<9}] {src:<15} -> {dst:<15} {extra:<25} {length} bytes")

    def print_summary(self):
        duration = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("Traffic Summary")
        print("=" * 80)

        print(f"Duration       : {duration:.2f} seconds")
        print(f"Total packets  : {self.total_packets}")
        print(f"Total bytes    : {self.total_bytes}")
        print(f"Average rate   : {self.total_packets / duration:.2f} packets/sec" if duration > 0 else "Average rate   : N/A")

        print("\nProtocols")
        for protocol, count in self.protocol_counter.most_common():
            print(f"  {protocol:<10} {count}")

        print("\nTop Source IPs")
        for ip, count in self.src_ip_counter.most_common(10):
            print(f"  {ip:<20} {count}")

        print("\nTop Destination IPs")
        for ip, count in self.dst_ip_counter.most_common(10):
            print(f"  {ip:<20} {count}")

        print("\nTop Destination Ports")
        for port, count in self.port_counter.most_common(10):
            print(f"  {port:<8} {count}")

        print("\nTop Conversations")
        for (src, dst), count in self.conversations.most_common(10):
            print(f"  {src:<15} -> {dst:<15} {count}")

        if self.dns_queries:
            print("\nDNS Queries")
            for query, count in self.dns_queries.most_common(10):
                print(f"  {query:<40} {count}")

        if self.http_hosts:
            print("\nHTTP Hosts")
            for host, count in self.http_hosts.most_common(10):
                print(f"  {host:<40} {count}")

        if self.http_paths:
            print("\nHTTP Paths")
            for path, count in self.http_paths.most_common(10):
                print(f"  {path:<40} {count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", default=None)
    parser.add_argument("-f", "--filter", default="ip")
    parser.add_argument("-c", "--count", type=int, default=0)
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    if args.list:
        for iface in get_if_list():
            print(iface)
        return

    analyzer = TrafficAnalyzer()

    print("Starting local traffic capture")
    print(f"Interface : {args.interface or 'default'}")
    print(f"Filter    : {args.filter}")
    print(f"Count     : {args.count if args.count > 0 else 'unlimited'}")
    print("Press Ctrl+C to stop\n")

    try:
        sniff(
            iface=args.interface,
            filter=args.filter,
            prn=analyzer.analyze_packet,
            store=False,
            count=args.count
        )
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.print_summary()

if __name__ == "__main__":
    main()

    
