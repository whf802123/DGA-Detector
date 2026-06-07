import argparse
import signal
import sys
import time
from collections import Counter
from scapy.all import sniff, IP, IPv6, TCP, UDP, ICMP, DNS, DNSQR, Raw, get_if_list


class UbuntuTrafficSniffer:
    def __init__(self):
        self.start_time = time.time()
        self.total_packets = 0
        self.total_bytes = 0
        self.protocols = Counter()
        self.src_ips = Counter()
        self.dst_ips = Counter()
        self.src_ports = Counter()
        self.dst_ports = Counter()
        self.connections = Counter()
        self.dns_queries = Counter()
        self.http_hosts = Counter()
        self.http_methods = Counter()
        self.http_paths = Counter()

    def process(self, packet):
        self.total_packets += 1
        self.total_bytes += len(packet)

        if IP in packet:
            ip = packet[IP]
            src_ip = ip.src
            dst_ip = ip.dst
            ip_version = "IPv4"
        elif IPv6 in packet:
            ip = packet[IPv6]
            src_ip = ip.src
            dst_ip = ip.dst
            ip_version = "IPv6"
        else:
            self.protocols["NON_IP"] += 1
            return

        self.src_ips[src_ip] += 1
        self.dst_ips[dst_ip] += 1

        protocol = "OTHER"
        detail = ""

        if TCP in packet:
            tcp = packet[TCP]
            sport = tcp.sport
            dport = tcp.dport

            self.src_ports[sport] += 1
            self.dst_ports[dport] += 1
            self.connections[(src_ip, sport, dst_ip, dport)] += 1

            if sport == 80 or dport == 80:
                protocol = "HTTP"
                self.parse_http(packet)
            elif sport == 443 or dport == 443:
                protocol = "HTTPS"
            elif sport == 22 or dport == 22:
                protocol = "SSH"
            else:
                protocol = "TCP"

            detail = f"{sport} -> {dport} flags={tcp.flags}"

        elif UDP in packet:
            udp = packet[UDP]
            sport = udp.sport
            dport = udp.dport

            self.src_ports[sport] += 1
            self.dst_ports[dport] += 1
            self.connections[(src_ip, sport, dst_ip, dport)] += 1

            if DNS in packet:
                protocol = "DNS"
                self.parse_dns(packet)
            elif sport == 123 or dport == 123:
                protocol = "NTP"
            elif sport == 5353 or dport == 5353:
                protocol = "MDNS"
            else:
                protocol = "UDP"

            detail = f"{sport} -> {dport}"

        elif ICMP in packet:
            protocol = "ICMP"
            detail = f"type={packet[ICMP].type} code={packet[ICMP].code}"

        self.protocols[protocol] += 1

        print(f"[{protocol:<6}] [{ip_version}] {src_ip:<39} -> {dst_ip:<39} {detail:<25} {len(packet)} bytes")

    def parse_dns(self, packet):
        if DNSQR in packet:
            name = packet[DNSQR].qname.decode(errors="ignore").rstrip(".")
            if name:
                self.dns_queries[name] += 1

    def parse_http(self, packet):
        if Raw not in packet:
            return

        payload = packet[Raw].load

        try:
            text = payload.decode("utf-8", errors="ignore")
        except Exception:
            return

        first_line = text.split("\r\n", 1)[0]
        parts = first_line.split()

        if len(parts) >= 2 and parts[0] in {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}:
            method = parts[0]
            path = parts[1]

            self.http_methods[method] += 1
            self.http_paths[path] += 1

            for line in text.split("\r\n"):
                if line.lower().startswith("host:"):
                    host = line.split(":", 1)[1].strip()
                    if host:
                        self.http_hosts[host] += 1
                    break

    def summary(self):
        duration = time.time() - self.start_time
        rate = self.total_packets / duration if duration > 0 else 0
        bandwidth = self.total_bytes / duration if duration > 0 else 0

        print("\n" + "=" * 100)
        print("Ubuntu Traffic Analysis Summary")
        print("=" * 100)

        print(f"Duration        : {duration:.2f} seconds")
        print(f"Total packets   : {self.total_packets}")
        print(f"Total bytes     : {self.total_bytes}")
        print(f"Packets/sec     : {rate:.2f}")
        print(f"Bytes/sec       : {bandwidth:.2f}")

        print("\nProtocols")
        for name, count in self.protocols.most_common():
            print(f"  {name:<10} {count}")

        print("\nTop Source IPs")
        for ip, count in self.src_ips.most_common(10):
            print(f"  {ip:<45} {count}")

        print("\nTop Destination IPs")
        for ip, count in self.dst_ips.most_common(10):
            print(f"  {ip:<45} {count}")

        print("\nTop Source Ports")
        for port, count in self.src_ports.most_common(10):
            print(f"  {port:<10} {count}")

        print("\nTop Destination Ports")
        for port, count in self.dst_ports.most_common(10):
            print(f"  {port:<10} {count}")

        print("\nTop Connections")
        for (src_ip, sport, dst_ip, dport), count in self.connections.most_common(10):
            print(f"  {src_ip}:{sport} -> {dst_ip}:{dport}  {count}")

        if self.dns_queries:
            print("\nDNS Queries")
            for query, count in self.dns_queries.most_common(10):
                print(f"  {query:<60} {count}")

        if self.http_hosts:
            print("\nHTTP Hosts")
            for host, count in self.http_hosts.most_common(10):
                print(f"  {host:<60} {count}")

        if self.http_methods:
            print("\nHTTP Methods")
            for method, count in self.http_methods.most_common():
                print(f"  {method:<10} {count}")

        if self.http_paths:
            print("\nHTTP Paths")
            for path, count in self.http_paths.most_common(10):
                print(f"  {path:<60} {count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", default=None)
    parser.add_argument("-f", "--filter", default="ip or ip6")
    parser.add_argument("-c", "--count", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for iface in get_if_list():
            print(iface)
        return

    sniffer = UbuntuTrafficSniffer()

    def stop_handler(signum, frame):
        sniffer.summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_handler)

    print("Ubuntu packet sniffer started")
    print(f"Interface : {args.interface or 'default'}")
    print(f"Filter    : {args.filter}")
    print(f"Count     : {args.count if args.count > 0 else 'unlimited'}")
    print("Press Ctrl+C to stop\n")

    sniff(
        iface=args.interface,
        filter=args.filter,
        prn=sniffer.process,
        store=False,
        count=args.count
    )

    sniffer.summary()

if __name__ == "__main__":
    main()

