# Cyber-ML-Detections



creat_syn_logs.py will create: 
fields: 
    src_ip, dest_ip, source.hostname/ destination.hostname, host.name, source.as.organization.name / destination.as.organization.name,
    source.geo*, destination.geo.*, destination.port, process.*, network.application, user.*, url.*





# Step 1: Obtain the Dataset
Recommended Dataset for Network Anomaly Detection
______________________________________________________
- CIC-IDS 2017
- URL: https://www.unb.ca/cic/datasets/ids-2017.html

- This dataset offers a comprehensive representation of realistic network traffic, encompassing various attack types such as botnets Distributed Denial of Service (DDoS) attacks, and brute-force intrusions.
- It includes essential network flow features, including Flow Bytes/s, Source IP, Destination IP, Protocol, and Flow Duration.
- Ideal for: Developing and evaluating solutions for inbound/outbound traffic volume anomaly detection.
# DISCLAIMER: I do not claim ownership of the CIC-IDS 2017 dataset; all intellectual property rights are reserved by its original creators. #

