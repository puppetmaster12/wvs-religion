import requests
import json

url = "http://localhost:8000/religion-descriptions"

religions = [
    "Adventist movement (Sunday observing)",
    "African traditional religions Ethnic",
    "Ahl-i Hadith",
    "Anglican",
    "Armenian Apostolic Church",
    "Baptist",
    "Brotherhood of the Cross and Star",
    "Buddhist",
    "Byzantine Rite",
    "Catholic",
    "Chinese traditional religion",
    "Church of Sweden",
    "Confucianists",
    "Cypriot Orthodox Church",
    "Druze",
    "Dutch Reformed Church",
    "Eastern Orthodox",
    "Ethiopian Orthodox Tewahedo Church",
    "Ethnic religions—excluding some in separate",
    "Evangelical",
    "Evangelical Church of the Augsburg Confession in",
    "Evangelical Lutheran Church of Brazil",
    "Evangelicalism",
    "Free Church",
    "Greek Byzantine Catholic Church",
    "Hindu",
    "Hoahaoism",
    "Iglesia ni Cristo (Church of Christ)",
    "Islam",
    "Jainism",
    "Jehovah's Witnesses",
    "Judaism",
    "Maronite Church",
    "Methodist",
    "Mormons",
    "Netherlands Reformed Churches",
    "Orthodox",
    "Orthodox Church of Greece",
    "Orthodox Church of Ukraine",
    "Other Christian",
    "Other syncretists",
    "Pentecostal and Charismatic",
    "Presbyterianism",
    "Protestant",
    "Protestant Churches without free churches",
    "Reformed Church in Romania",
    "Roman Catholic; Latin Church;",
    "Russian Orthodox Church",
    "Serbian Orthodox Church",
    "Seventh-day Adventist Church",
    "Shia",
    "Sikhism",
    "Slovak Greek Catholic Church",
    "Spiritism",
    "Sunni",
    "Taoism (Han Chinese)",
    "The African Church",
    "United Evangelical Church—in Nigeria",
    "Vietnam Ancestral worshipping / Tradition",
    "Vodou",
    "Yiguandao",
    "Zion Christian Church"
]

headers = {
    "Content-Type": "application/json",
}

payload = {"religions": religions}

response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    with open("api_response.json", "w", encoding="utf-8") as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)
    print("Descriptions saved.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)