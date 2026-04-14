"""
Generate the Nexora Industries demo PDF for the Hierarchical Graph RAG presentation.

Creates a richly-structured fictional enterprise narrative -- a completely
original story with clear entities, relationships, and cross-references that
the LLM must discover organically (no numbered sections or headings).

The continuous text covers:
  - Company founding and early history
  - Key people (CEO, CTO, department heads -- all fictional)
  - Products (Pulse CRM, Athena AI assistant, Vortex analytics)
  - Departments (R&D, Sales, Operations, HR, Security)
  - Acquisitions and partnerships
  - Internal incidents and crises
  - Competitive dynamics
  - Strategic pivots and future plans

Entities overlap naturally across paragraphs to produce rich graph structure.

Usage:
    python gen_pdfs.py
"""

import os
from fpdf import FPDF


# ── Nexora Industries -- Continuous Narrative ────────────────────────────────
# NOTE: All text uses ASCII-safe characters only (no em-dashes, curly quotes)
# to avoid latin-1 encoding errors with fpdf's built-in Helvetica font.
#
# DESIGN RATIONALE: This is written as a single continuous narrative without
# section headings. Entities (people, products, departments, events) are
# woven throughout so the LLM must discover communities and concepts itself.

NEXORA_NARRATIVE = """\
Nexora Industries was founded in 2016 by Evelyn Hartwell and Marcus Tan in \
a small office in Austin, Texas. Evelyn, a former product lead at Salesforce, \
had grown frustrated with the rigid design of enterprise CRM systems that \
forced sales teams into rigid workflows instead of adapting to how people \
actually sell. Marcus, a machine learning researcher who had spent six years \
at Google Brain working on natural language understanding, believed that \
conversational AI could fundamentally change how businesses interact with \
their own data. Together they drafted the founding vision for Nexora: build \
an intelligent enterprise platform that learns from every user interaction \
and reshapes itself around how teams actually work.

Their first product, Pulse, launched in early 2017 as a lightweight CRM \
aimed at mid-market sales teams. Pulse differentiated itself from Salesforce \
and HubSpot by using a recommendation engine that analyzed email threads, \
calendar events, and deal notes to automatically suggest next-best actions \
for each sales representative. Within eighteen months, Pulse had attracted \
over two thousand paying customers and generated twelve million dollars in \
annual recurring revenue. The early traction caught the attention of \
Meridian Ventures, a Silicon Valley venture capital firm that led Nexora's \
Series A round of twenty-eight million dollars in mid-2018. Priya Nair, a \
partner at Meridian Ventures who specialized in enterprise SaaS, joined \
Nexora's board of directors and pushed the company to expand beyond CRM \
into a broader enterprise intelligence platform.

By late 2018, Nexora had grown to one hundred and twenty employees organized \
into five departments. The Research and Development department, led by Chief \
Technology Officer Marcus Tan, focused on core AI capabilities including \
natural language processing, predictive analytics, and knowledge graph \
construction. The Sales department, headed by Vice President of Sales Jordan \
Castillo, managed the growing base of Pulse customers and pursued enterprise \
deals. The Operations department, run by Chief Operating Officer Sandra \
Levine, handled infrastructure, cloud deployment on Amazon Web Services, \
and customer onboarding. The Human Resources department, directed by \
Olivia Mendes, was responsible for recruiting engineering talent in a highly \
competitive Austin tech market. Finally, the Security and Compliance team, \
led by Chief Information Security Officer Raj Malhotra, ensured that \
customer data handled by Pulse met SOC 2 Type II and GDPR standards.

In January 2019, Marcus Tan's R&D team began developing Athena, an AI-powered \
virtual assistant designed to sit on top of the Pulse CRM and answer natural \
language questions about sales pipelines, customer histories, and revenue \
forecasts. Athena used a retrieval-augmented generation architecture that \
combined a proprietary knowledge graph with a fine-tuned large language model. \
The knowledge graph, internally code-named Cortex, ingested data from Pulse \
CRM records, Slack conversations, Jira tickets, and Confluence documentation \
to build a unified representation of each customer's organizational knowledge. \
Marcus believed that Cortex was Nexora's most defensible technology asset \
because the knowledge graph improved continuously as more data flowed through it.

Athena entered private beta in June 2019 with fifteen pilot customers. The \
results were impressive but uneven. Customers with clean, well-structured \
data in Pulse saw Athena answer pipeline questions with over ninety percent \
accuracy. However, customers with messy or incomplete CRM data experienced \
frequent hallucinations where Athena fabricated deal amounts or invented \
customer contacts that did not exist. Jordan Castillo's sales team reported \
that prospects loved the Athena demos but worried about trusting an AI \
assistant with mission-critical sales decisions. This tension between Athena's \
potential and its reliability became a central strategic debate inside Nexora \
throughout 2019 and 2020.

In September 2019, Nexora acquired a small Boston-based startup called \
DataMesh Labs for fourteen million dollars. DataMesh Labs had developed a \
data quality engine called Sentinel that could automatically detect anomalies, \
duplicates, and missing fields in enterprise datasets. The acquisition was \
championed by Sandra Levine, who argued that integrating Sentinel into the \
Pulse and Athena pipeline would solve the data quality problems plaguing \
Athena's accuracy. The DataMesh Labs team of twelve engineers, led by founder \
Chen Wei, relocated to Austin and was absorbed into the R&D department under \
Marcus Tan. Chen Wei took the title of Vice President of Data Engineering.

The integration of Sentinel into Athena took longer than expected. By March \
2020, the Sentinel data cleaning pipeline was running in production but had \
introduced a new problem: it was so aggressive at flagging uncertain data \
that it sometimes quarantined legitimate customer records, causing Pulse \
users to see gaps in their dashboards. Raj Malhotra's security team \
discovered that the quarantine rules interacted badly with the GDPR data \
retention policies, creating situations where customer data was simultaneously \
flagged for deletion by GDPR workflows and flagged for review by Sentinel. \
Resolving these conflicts required a four-month engineering sprint led jointly \
by Chen Wei and Raj Malhotra that produced a unified data governance \
framework called TrustLayer. TrustLayer became the backbone of Nexora's \
compliance architecture and was later cited by several enterprise customers \
as a key reason for choosing Nexora over competitors.

While Nexora wrestled with internal technical challenges, the competitive \
landscape was shifting. Orion Systems, a larger enterprise software company \
based in Seattle, launched a competing AI assistant called Argos in January \
2020. Argos was more polished in its user interface and had deeper \
integrations with Microsoft 365 and Google Workspace. However, Argos relied \
on a simpler retrieval architecture without a knowledge graph, which meant \
it struggled with multi-hop reasoning questions like "Which deals closed by \
Jordan's team last quarter involved contacts who also attended the Chicago \
trade show?" Nexora's Athena, powered by the Cortex knowledge graph, could \
answer such questions naturally by traversing entity relationships.

Jordan Castillo's sales team used this knowledge-graph advantage as the \
primary differentiator in competitive deals against Orion Systems. In a \
pivotal enterprise deal with Falkner Manufacturing, a five-hundred-employee \
industrial company based in Detroit, Jordan's team demonstrated Athena's \
ability to trace relationships across Falkner's supply chain data and \
surface insights that Argos could not match. Falkner Manufacturing signed \
a three-year contract worth one point two million dollars, and their CTO \
Rebecca Torres became one of Athena's most vocal public advocates, speaking \
at two industry conferences about how the knowledge graph approach \
transformed their sales operations.

In mid-2020, Evelyn Hartwell decided to pursue a major strategic pivot. She \
wanted Nexora to evolve from a CRM-focused company into a horizontal \
enterprise intelligence platform. The plan, internally called Project \
Horizon, involved building a new product called Vortex -- an analytics \
and decision intelligence layer that could ingest data from any enterprise \
source (ERP systems, supply chain databases, HR platforms, financial \
systems) and use the Cortex knowledge graph to provide cross-functional \
insights. Evelyn believed that Vortex would expand Nexora's total \
addressable market from three billion dollars in CRM to over fifteen \
billion dollars in enterprise analytics.

Project Horizon was controversial inside Nexora. Marcus Tan supported the \
vision but warned that building Vortex would require significant \
investment in the Cortex knowledge graph, which would need new data \
connectors, improved entity resolution, and a more scalable graph database \
infrastructure. Sandra Levine expressed concern that the operations team \
was already stretched thin supporting Pulse and Athena, and adding a third \
product would overwhelm their customer success capacity. Jordan Castillo \
argued that the sales team should focus on closing more Pulse and Athena \
deals to hit revenue targets rather than diverting attention to an unproven \
new product. Despite these objections, Evelyn secured board approval for \
Project Horizon, with Priya Nair casting the deciding vote at the September \
2020 board meeting.

To lead the Vortex development effort, Evelyn recruited Tomoko Ishida, a \
former engineering director at Palantir who had extensive experience building \
large-scale data integration platforms. Tomoko joined Nexora in November 2020 \
as Vice President of Product and was given a dedicated team of twenty \
engineers pulled from the R&D department. This created friction with Marcus \
Tan, who felt that his team was being raided. The tension between Marcus \
and Tomoko became a recurring management challenge that Evelyn had to \
mediate throughout 2021.

Vortex entered alpha testing in April 2021 with three design partners: \
Falkner Manufacturing, Bridgewater Logistics (a Chicago-based freight \
company), and Cascadia Health Systems (a hospital network in Portland, \
Oregon). The early results showed that Vortex excelled at cross-referencing \
data from different enterprise systems. At Bridgewater Logistics, Vortex \
discovered that their highest-margin shipping routes overlapped with routes \
that had the most frequent customer complaints, a correlation that \
Bridgewater's own analytics team had missed for years. At Cascadia Health \
Systems, Vortex identified patterns in patient scheduling data that \
revealed bottlenecks in the emergency department caused by staffing \
mismatches, leading to a twelve percent improvement in patient throughput \
after Cascadia adjusted their nurse schedules based on Vortex's recommendations.

However, the Vortex alpha also exposed serious scaling problems. The Cortex \
knowledge graph, which had been designed to handle CRM-scale data volumes, \
struggled under the weight of ERP and healthcare datasets that were orders \
of magnitude larger. Graph query latencies increased from milliseconds to \
seconds, making the interactive analytics experience unacceptably slow. \
Marcus Tan proposed migrating Cortex from its existing Neo4j-based \
architecture to a distributed graph database, but Chen Wei argued that the \
performance problems were caused by inefficient entity resolution rather \
than database limitations. The two engineers spent three months debating \
approaches before settling on a hybrid solution: they would optimize entity \
resolution using Chen Wei's improved Sentinel algorithms while also \
implementing graph partitioning to distribute query load across multiple \
Neo4j instances.

In August 2021, Nexora faced its most serious crisis. A security researcher \
named Anika Svensson discovered a vulnerability in Athena's API that allowed \
unauthorized users to access Cortex knowledge graph data belonging to other \
tenants. The vulnerability, which Raj Malhotra's security team designated \
as Incident Zephyr, affected approximately thirty enterprise customers \
and exposed metadata about their CRM records, though no raw customer data \
was directly leaked. Raj Malhotra led the incident response, coordinating \
with Nexora's legal team and the affected customers. The engineering fix \
took seventy-two hours and required a complete redesign of Athena's \
multi-tenant isolation layer. Evelyn Hartwell personally called every \
affected customer CEO to apologize and offer extended service credits.

Incident Zephyr had lasting consequences. Three customers, including \
Bridgewater Logistics, temporarily suspended their Nexora contracts pending \
a third-party security audit. Jordan Castillo reported that at least five \
prospective enterprise deals were lost directly because of the breach. \
Internally, the incident accelerated the development of TrustLayer v2, an \
enhanced security framework that added end-to-end encryption for all Cortex \
graph data and implemented zero-trust access controls. Raj Malhotra hired \
eight additional security engineers and established a dedicated red team \
that continuously tested Nexora's products for vulnerabilities. The \
investment in security proved strategically valuable: within a year, Nexora \
achieved FedRAMP authorization, opening the door to federal government \
contracts.

By early 2022, Nexora had grown to three hundred and fifty employees with \
annual recurring revenue of forty-five million dollars. Pulse remained the \
core revenue driver, but Athena was gaining traction among enterprise \
customers who valued the knowledge graph capabilities. Vortex was still in \
beta but showed strong early metrics. Meridian Ventures led Nexora's \
Series B round of sixty-five million dollars, at a valuation of four \
hundred million dollars. Priya Nair publicly stated that Nexora represented \
the future of enterprise AI because of its unique combination of a \
knowledge graph foundation and a data quality layer.

The competitive battle with Orion Systems intensified in 2022. Orion \
responded to Nexora's knowledge graph advantage by acquiring GraphLoom, a \
San Francisco startup specializing in enterprise knowledge graphs, for \
fifty-two million dollars. The GraphLoom acquisition gave Orion's Argos \
product knowledge graph capabilities that narrowed the gap with Athena. \
Jordan Castillo's sales team noticed that competitive win rates against \
Orion dropped from seventy percent to fifty-five percent in the second \
half of 2022. In response, Evelyn Hartwell authorized an accelerated \
investment in Athena's reasoning capabilities, directing Marcus Tan to \
integrate chain-of-thought prompting and multi-step retrieval into \
Athena's answer generation pipeline.

Olivia Mendes, the HR director, faced a parallel challenge in 2022. The \
Great Resignation had made it extremely difficult to retain senior \
engineers, and Nexora lost seventeen engineers to competitors including \
Orion Systems, Google, and several AI startups. Olivia implemented a \
retention program called Catalyst that included equity refreshes, \
flexible remote work policies, and an internal innovation program where \
engineers could spend twenty percent of their time on self-directed \
research projects. The Catalyst program reduced attrition from twenty-two \
percent to nine percent over twelve months and produced several innovations \
that were incorporated into Nexora's products, including an improved vector \
search algorithm developed by a junior engineer named Leo Park during his \
Catalyst project time.

In 2023, Tomoko Ishida led the general availability launch of Vortex. The \
product was positioned as an enterprise intelligence platform for mid-market \
and enterprise companies that needed cross-functional analytics without \
building a dedicated data science team. Vortex's launch was supported by \
a strategic partnership with Snowflake, allowing Vortex to directly query \
data warehouses without requiring data extraction. The partnership was \
negotiated by Sandra Levine, who had cultivated relationships with \
Snowflake's enterprise team over several years. The Snowflake integration \
became Vortex's most popular feature, used by over sixty percent of Vortex \
customers within the first six months.

Looking toward 2024 and beyond, Evelyn Hartwell outlined Nexora's next \
strategic initiative, internally called Project Aurora. Aurora aimed to \
add real-time collaboration features to the Cortex knowledge graph, \
allowing multiple teams across an organization to simultaneously contribute \
to and query a shared intelligence layer. Marcus Tan's R&D team began \
experimenting with federated learning techniques that would allow Cortex \
to learn from customer data patterns without centralizing sensitive \
information. The Aurora initiative represented Nexora's bet that the \
future of enterprise software lay not in isolated applications but in \
connected intelligence networks where knowledge flows seamlessly across \
organizational boundaries. Tomoko Ishida was named Chief Product Officer \
to oversee the unified product strategy across Pulse, Athena, and Vortex, \
while Marcus Tan retained his CTO role with a renewed focus on fundamental \
AI research and the Cortex knowledge graph platform that underpinned \
everything Nexora built.\
"""


def create_pdf(filename: str, title: str, text: str):
    """Create a PDF with continuous flowing text (no section headings)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Continuous body text -- no headings, just paragraphs
    pdf.set_font("Helvetica", "", 10)
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            pdf.multi_cell(0, 5, paragraph)
            pdf.ln(3)

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    pdf.output(out_path)
    print(f"  Created: {out_path}")


def main():
    print("Generating Nexora Industries demo PDF...")
    create_pdf(
        "nexora_industries.pdf",
        "Nexora Industries: Enterprise Intelligence Platform",
        NEXORA_NARRATIVE,
    )
    print("Done! PDF saved to data/nexora_industries.pdf")


if __name__ == "__main__":
    main()
