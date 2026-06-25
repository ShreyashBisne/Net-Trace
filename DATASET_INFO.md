# AI-Driven IDS Dataset Information

## Dataset Overview

The AI-Driven Intrusion Detection System uses the **NSL-KDD dataset**, which is an enhanced version of the widely-used KDD'99 dataset. This dataset has been specifically designed to benchmark and evaluate intrusion detection systems.

## Dataset Files

### Training Data
- **File**: `train.csv`
- **Records**: 125,973 network traffic records
- **Purpose**: Used to train the machine learning models (Random Forest, Neural Network, Logistic Regression)

### Testing Data
- **File**: `test.csv`
- **Records**: 22,544 network traffic records
- **Purpose**: Used to evaluate model performance and accuracy

## Feature Description

The dataset contains **43 columns** representing different network traffic characteristics:

### Basic Connection Features (4 features)
1. **duration** - Length of the connection in seconds
2. **protocol_type** - Protocol used (tcp, udp, icmp)
3. **service** - Network service (http, ftp, smtp, ssh, dns, etc.)
4. **flag** - Normal or error status of the connection (SF, S0, S1, S2, S3, REJ, RSTO, RSTOS0, RSTR, SH, SHR, OTH)

### Content Features (12 features)
5. **src_bytes** - Number of data bytes from source to destination
6. **dst_bytes** - Number of data bytes from destination to source
7. **land** - 1 if connection is from/to the same host/port; 0 otherwise
8. **wrong_fragment** - Number of "wrong" fragments
9. **urgent** - Number of urgent packets
10. **hot** - Number of "hot" indicators
11. **num_failed_logins** - Number of failed login attempts
12. **logged_in** - 1 if successfully logged in; 0 otherwise
13. **num_compromised** - Number of "compromised" conditions
14. **root_shell** - 1 if root shell is obtained; 0 otherwise
15. **su_attempted** - 1 if "su" command attempted; 0 otherwise
16. **num_root** - Number of "root" accesses

### Host-Based Traffic Features (8 features)
17. **num_file_creations** - Number of file creation operations
18. **num_shells** - Number of shell prompts
19. **num_access_files** - Number of operations on access control files
20. **num_outbound_cmds** - Number of outbound commands in an ftp session
21. **is_host_login** - 1 if the login belongs to the "host" list; 0 otherwise
22. **is_guest_login** - 1 if the login is a "guest" login; 0 otherwise
23. **count** - Number of connections to the same host as the current connection in the past two seconds
24. **srv_count** - Number of connections to the same service as the current connection in the past two seconds

### Network Traffic Statistics (11 features)
25. **serror_rate** - Percentage of connections that have "SYN" errors
26. **srv_serror_rate** - Percentage of connections that have "SYN" errors to the same service
27. **rerror_rate** - Percentage of connections that have "REJ" errors
28. **srv_rerror_rate** - Percentage of connections that have "REJ" errors to the same service
29. **same_srv_rate** - Percentage of connections to the same service
30. **diff_srv_rate** - Percentage of connections to different services
31. **srv_diff_host_rate** - Percentage of connections to different hosts for the same service
32. **dst_host_count** - Number of connections having the same destination host
33. **dst_host_srv_count** - Number of connections having the same destination host and service
34. **dst_host_same_srv_rate** - Percentage of connections having the same destination host and service
35. **dst_host_diff_srv_rate** - Percentage of connections having the same destination host but different service

### Additional Statistics (6 features)
36. **dst_host_same_src_port_rate** - Percentage of connections having the same source port
37. **dst_host_srv_diff_host_rate** - Percentage of connections having the same destination host and service but different source host
38. **dst_host_serror_rate** - Percentage of connections with SYN errors to the same destination host
39. **dst_host_srv_serror_rate** - Percentage of connections with SYN errors to the same destination host and service
40. **dst_host_rerror_rate** - Percentage of connections with REJ errors to the same destination host
41. **dst_host_srv_rerror_rate** - Percentage of connections with REJ errors to the same destination host and service

### Target Variables (2 features)
42. **attack** - Type of attack (normal, back, buffer_overflow, ftp_write, guess_passwd, etc.)
43. **difficulty_level** - Difficulty level of the record (1-21)

## Label Distribution

### Training Set
- **Normal Traffic**: ~67,343 records (53.5%)
- **Attack Traffic**: ~58,630 records (46.5%)

### Test Set
- **Normal Traffic**: ~9,711 records (43.1%)
- **Attack Traffic**: ~12,833 records (56.9%)

## Attack Types in Dataset

The dataset includes various attack types categorized as:

1. **Denial of Service (DoS)** - back, land, neptune, pod, smurf, teardrop
2. **User to Root (U2R)** - buffer_overflow, loadmodule, perl, rootkit
3. **Remote to Local (R2L)** - ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
4. **Probing** - ipsweep, nmap, portsweep, satan

## Model Training

The application trains three machine learning models:

1. **Random Forest Classifier** - 100 estimators
2. **Neural Network (MLP)** - Hidden layers: (64, 32)
3. **Logistic Regression** - Max iterations: 1000

### Preprocessing Steps
- **Scaling**: StandardScaler normalization applied to all features
- **Encoding**: Label encoding for categorical features (protocol_type, service, flag)
- **Binary Classification**: Attack (1) vs Normal (0)

## Data Source

The NSL-KDD dataset was obtained from:
- **Original Source**: University of New Brunswick (UNB)
- **GitHub Repository**: https://github.com/jmnwong/NSL-KDD-Dataset
- **Publication**: "NSL-KDD: A Benchmark Dataset for Network Intrusion Detection Systems"

## Key Advantages of NSL-KDD

1. **No Redundant Records** - Training set is free from duplicate records
2. **No Duplicate Test Records** - Test set has no duplicates
3. **Balanced Difficulty Levels** - Records selected inversely proportional to difficulty
4. **Comprehensive Features** - 41 network traffic features + 2 label features
5. **Realistic Representation** - Better representation of real network traffic patterns

## Usage in Application

The application uses this dataset to:
- Display statistical dashboards of network traffic patterns
- Train and evaluate intrusion detection models
- Perform single-record predictions
- Conduct batch analysis on multiple records
- Provide model performance metrics and comparisons

## Data Privacy and Ethics

This is a benchmark dataset for research and educational purposes. It contains no personally identifiable information (PII) and is suitable for academic research, model development, and system evaluation.
