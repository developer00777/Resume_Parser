"""Generate a sample resume PDF for testing."""

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError:
    print("Installing reportlab...")
    import subprocess
    subprocess.check_call(["pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

import os

OUTPUT = os.path.join(os.path.dirname(__file__), "sample_resume.pdf")

LINES = [
    ("RAJESH KUMAR SHARMA", 16, True),
    ("Senior Software Engineer", 12, False),
    ("", 10, False),
    ("Email: rajesh.sharma@gmail.com | Phone: +91-9876543210 | Alt: +91-11-26543210", 9, False),
    ("Location: Bangalore, Karnataka, India", 9, False),
    ("", 10, False),
    ("PROFESSIONAL SUMMARY", 12, True),
    ("Experienced Senior Software Engineer with 6+ years of expertise in building", 9, False),
    ("scalable backend systems, microservices architecture, and cloud-native applications.", 9, False),
    ("Proficient in Python, Go, and JavaScript with strong focus on performance optimization.", 9, False),
    ("", 10, False),
    ("SKILLS", 12, True),
    ("Python, Go, JavaScript, TypeScript, FastAPI, Django, Flask, React, Node.js,", 9, False),
    ("PostgreSQL, MongoDB, Redis, Docker, Kubernetes, AWS, GCP, Terraform, CI/CD,", 9, False),
    ("Git, REST APIs, GraphQL, Kafka, RabbitMQ, Elasticsearch, Linux, Agile/Scrum", 9, False),
    ("", 10, False),
    ("WORK EXPERIENCE", 12, True),
    ("Senior Software Engineer | Flipkart | Jan 2022 - Present", 10, True),
    ("- Designed high-throughput order processing microservices (50K+ req/min)", 9, False),
    ("- Led monolith to microservices migration using Kubernetes (-70% deploy time)", 9, False),
    ("- Implemented real-time inventory tracking with Kafka and Redis", 9, False),
    ("- Mentored 4 junior developers and conducted code reviews", 9, False),
    ("", 10, False),
    ("Software Engineer | Infosys | Jul 2019 - Dec 2021", 10, True),
    ("- Built RESTful APIs using Python/Django for enterprise banking clients", 9, False),
    ("- Developed automated testing framework (45% -> 85% code coverage)", 9, False),
    ("- Optimized database queries reducing response time by 40%", 9, False),
    ("", 10, False),
    ("Junior Software Developer | TCS | Jun 2017 - Jun 2019", 10, True),
    ("- Developed internal tools and dashboards using Flask and React", 9, False),
    ("- Wrote unit and integration tests for backend services", 9, False),
    ("", 10, False),
    ("EDUCATION", 12, True),
    ("B.Tech in Computer Science and Engineering", 10, True),
    ("Indian Institute of Technology (IIT), Hyderabad | 2017 | CGPA: 8.7/10", 9, False),
    ("", 10, False),
    ("Higher Secondary (XII) - CBSE Board", 10, True),
    ("Delhi Public School, New Delhi | 2013 | 92.4%", 9, False),
    ("", 10, False),
    ("CERTIFICATIONS", 12, True),
    ("- AWS Certified Solutions Architect - Associate (2023)", 9, False),
    ("- Certified Kubernetes Administrator (CKA) (2022)", 9, False),
    ("- Google Cloud Professional Cloud Architect (2021)", 9, False),
]


def generate():
    c = canvas.Canvas(OUTPUT, pagesize=A4)
    width, height = A4
    y = height - 50

    for text, size, bold in LINES:
        if y < 50:
            c.showPage()
            y = height - 50
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.drawString(50, y, text)
        y -= size + 4

    c.save()
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    generate()
