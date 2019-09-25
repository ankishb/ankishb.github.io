
## Types of Spam Filters
As a message traverses from the sender to the subscriber’s inbox, various types of filters can influence deliverability and inbox placement:
1. Gateway spam filters:
    - physical servers authentication
    - IITK server have some authentication check for email service
    - IP address check, cisco ironport, etc
2. Third Party spam filter:
    - App provider
    - gmail/yahoo/outlook etc. have some spam filtering check
        1. `Content filters` – review the content within a message
        2. `Header filters` – review the email header in search of falsified information
        3. `General blacklist filters` – stop all emails that come from a blacklisted file of known spammers
        4. `Rules-based filters` – use user-defined criteria – such as specific senders or specific wording in the subject line or body – to block spam
        5. `Permission filters` – require anyone sending a message to be pre-approved by the recipient
        6. `Challenge-response filters` – require anyone sending a message to enter a code in order to gain permission to send email
    - Examples of third party filters spam filters include Cloudmark and MessageLabs.
3. Desktop spam filters:
    - Can be classified under type 2 as well
    - our customized filter
    - feeback based filter 
    - Outlook uses Microsoft’s anti-spam filter SmartScreen to help filter email



## Yahoo japan
- It is different from yahoo, it is founded by yahoo and soft-bank to manage web search engine for japan coutry.
- `80%` people in japan use yahoo internet services
- Have rank 28 in Alexa’s Top 500 Sites on the Web
- 68 billion pageviews per month (2018). 
- Internet Giant Needs to Manage Growing Storage Needs While Reducing Cost and Operational Overhead
- It sought to evolve from a “smartphone company” to a “data company,” Yahoo! JAPAN was challenged to expand its capabilities to leverage vast data stores, including demographic, psychographic, e-commerce, real-time search, and web browsing.

The company operates more than 75,000 physical servers and over 120,000 virtual machines across its six data centers in Japan and one in the United States, with more than 60 PB of storage system capacity total. Many of its services run on Yahoo! JAPAN’s private cloud system, which are built on more than 70 OpenStack clusters. These clusters are operated by a team of fewer than 20 engineers.

The problem – Finding the best way to manage growing storage needs while limiting the acquisition and operational expenses that scaling resources require.

In managing clusters of this scale, Yahoo! JAPAN is always looking at ways to reduce operational costs. Key to achieving this goal is implementing solutions that improve efficiency, such as developing a chatbot to help automate support processes. Yahoo! JAPAN has also centralized its system logs and metrics to allow staff to monitor and visualize the system through a single dashboard.


## regular expression for `telephone number`
- ^\s*?\d{3}-\d{3}-\d{4}\s*?