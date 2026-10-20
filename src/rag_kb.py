"""
rag_kb.py — NephroSense Retrieval-Augmented Knowledge Base
===========================================================
How it works
------------
1. Every KB article has:
   - A title
   - Keywords in English, Hindi (romanised + Devanagari), and Marathi
   - A concise medical summary (English — sent to the LLM only for English queries)
   - A verified source URL from KDIGO, NKF, WHO, or NIH
   - Marathi and Hindi translations of the summary (sent to LLM for those languages)

2. Retrieval (retrieve_context):
   - Tokenise the query into words
   - Score each article by keyword overlap (exact + prefix match)
   - Return top-k articles including their source URLs
   - The LLM receives the relevant article content + source links as context

3. Source attribution:
   - Every answer that uses a KB article includes a "Source: [name](url)" line
   - If no article matches, the LLM answers from its own training knowledge
     and the reply is labelled "General AI knowledge — confirm with a doctor"

4. Medical accuracy:
   - All articles are sourced from KDIGO 2024, NKF, WHO, or NIH MedlinePlus
   - The LLM is instructed never to invent facts not in the context
   - Emergency detection happens before any LLM call

Why no vector database?
   - Adding ChromaDB/FAISS would require pip install + model download (~500MB)
     just for 18 articles — overkill and a fragile dependency on a patient device.
   - TF-IDF style keyword scoring is transparent, debuggable, and instant.
   - The LLM (Groq llama-3.3-70b or Gemini 2.0 Flash) has strong enough
     medical knowledge that precise vector retrieval over 18 articles adds
     no measurable benefit.
"""
from __future__ import annotations
import re
from typing import NamedTuple


class KBArticle(NamedTuple):
    title_en: str
    title_mr: str
    title_hi: str
    keywords: list[str]          # lower-case; checked against query
    summary_en: str              # sent to LLM for English queries
    summary_mr: str              # sent to LLM for Marathi queries
    summary_hi: str              # sent to LLM for Hindi queries
    source_name: str             # human-readable citation
    source_url: str              # verified URL


ARTICLES: list[KBArticle] = [

    KBArticle(
        title_en="Chronic Kidney Disease (CKD) — Stages & Overview",
        title_mr="तीव्र मूत्रपिंड आजार (CKD) — अवस्था आणि माहिती",
        title_hi="क्रोनिक किडनी रोग (CKD) — चरण और जानकारी",
        keywords=[
            "ckd", "chronic kidney", "kidney disease", "kidney failure", "kidney damage",
            "stage", "egfr", "gfr", "creatinine high", "kidney function",
            "किडनी", "मूत्रपिंड", "किडनी खराब", "किडनी आजार", "kidney",
            "gudda", "gurde", "gurda", "gurdaon",
        ],
        summary_en=(
            "CKD means the kidneys have been damaged or working poorly for 3+ months. "
            "Stages by eGFR: G1 ≥90 (damage but normal function), G2 60–89, "
            "G3a 45–59, G3b 30–44, G4 15–29, G5 <15 (kidney failure). "
            "Early CKD usually has NO symptoms — regular blood and urine tests are essential. "
            "Main causes: diabetes, high blood pressure, glomerulonephritis, PKD, long-term NSAIDs. "
            "Control blood sugar and BP, avoid NSAID painkillers, eat less salt, stop smoking."
        ),
        summary_mr=(
            "CKD म्हणजे ३ किंवा अधिक महिन्यांपासून मूत्रपिंड खराब झालेले किंवा कमी काम करणारे. "
            "eGFR नुसार टप्पे: G1 (≥90, नुकसान पण काम सामान्य), G2 (60–89), "
            "G3a (45–59), G3b (30–44), G4 (15–29), G5 (<15 — मूत्रपिंड निकामी). "
            "सुरुवातीच्या CKD मध्ये सहसा लक्षणे नसतात — रक्त आणि लघवीच्या तपासण्या महत्त्वाच्या. "
            "मुख्य कारणे: मधुमेह, उच्च रक्तदाब, NSAIDs वेदनाशामक. "
            "रक्तातील साखर आणि रक्तदाब नियंत्रणात ठेवा, मीठ कमी खा."
        ),
        summary_hi=(
            "CKD का मतलब है कि किडनी ३ या अधिक महीनों से खराब हो रही है। "
            "eGFR के अनुसार चरण: G1 (≥90), G2 (60–89), G3a (45–59), G3b (30–44), G4 (15–29), G5 (<15 — किडनी फेलियर)। "
            "शुरुआती CKD में अक्सर कोई लक्षण नहीं होते — खून और पेशाब की जांच जरूरी है। "
            "मुख्य कारण: मधुमेह, उच्च रक्तचाप, NSAID दर्दनिवारक। "
            "रक्त शर्करा और BP नियंत्रित रखें, नमक कम खाएं।"
        ),
        source_name="KDIGO 2024 CKD Guidelines",
        source_url="https://kdigo.org/guidelines/ckd-evaluation-and-management/",
    ),

    KBArticle(
        title_en="Kidney Stones",
        title_mr="मूत्रपिंडातील खडे (किडनी स्टोन)",
        title_hi="किडनी स्टोन (पथरी)",
        keywords=[
            "kidney stone", "stones", "stone", "calculi", "renal colic", "urolithiasis",
            "mutkhada", "muthkhada", "pathri", "पथरी", "खडे", "मुतखडा",
            "flank pain", "groin pain", "blood in urine", "urine blood",
            "stone symptoms", "kidney stone symptoms", "stone pain",
        ],
        summary_en=(
            "Kidney stones are hard deposits of minerals that form in the kidney. "
            "Symptoms: sudden severe pain in the side or lower back spreading to groin, "
            "blood in urine, nausea, burning urination. "
            "Treatment: drink 2.5–3 litres of water daily; small stones often pass on their own. "
            "See a doctor urgently for severe pain, fever with pain, or inability to pass urine. "
            "Prevention: less salt, less animal protein, normal calcium intake, stay well hydrated."
        ),
        summary_mr=(
            "किडनी स्टोन म्हणजे मूत्रपिंडात तयार होणारे खनिजांचे कठीण खडे. "
            "लक्षणे: कंबरेत किंवा बाजूला तीव्र वेदना जी मांडीपर्यंत जाते, लघवीत रक्त, मळमळ. "
            "उपाय: दिवसातून २.५–३ लिटर पाणी प्या; लहान खडे स्वतः निघतात. "
            "तीव्र वेदना, ताप किंवा लघवी बंद झाल्यास तातडीने डॉक्टरांकडे जा. "
            "प्रतिबंध: मीठ कमी, जास्त पाणी, जनावरांचे मांस कमी."
        ),
        summary_hi=(
            "किडनी स्टोन (पथरी) किडनी में बनने वाले खनिज के कठोर टुकड़े होते हैं। "
            "लक्षण: कमर या बाजू में अचानक तेज दर्द जो पेट के नीचे तक जाए, पेशाब में खून, मतली। "
            "उपाय: रोजाना 2.5–3 लीटर पानी पिएं; छोटी पथरी खुद निकल जाती है। "
            "तेज दर्द, बुखार या पेशाब बंद होने पर तुरंत डॉक्टर के पास जाएं। "
            "बचाव: कम नमक, खूब पानी, कम मांस।"
        ),
        source_name="National Kidney Foundation — Kidney Stones",
        source_url="https://www.kidney.org/kidney-topics/kidney-stones",
    ),

    KBArticle(
        title_en="Urinary Tract Infection (UTI)",
        title_mr="मूत्रमार्गाचा संसर्ग (UTI)",
        title_hi="मूत्र मार्ग संक्रमण (UTI)",
        keywords=[
            "uti", "urine infection", "urinary infection", "burning urine", "burning urination",
            "pyelonephritis", "cystitis", "painful urination", "frequent urination",
            "जळजळ", "जलन", "लघवीत जळजळ", "पेशाब में जलन", "mutravata",
        ],
        summary_en=(
            "UTI causes burning or urgent urination, frequent urination, lower belly pain, "
            "cloudy or smelly urine. If infection spreads to kidney (pyelonephritis): fever, chills, "
            "back or flank pain, vomiting — needs immediate medical attention. "
            "Diagnosis: urine test (dipstick + culture). Treatment: doctor-prescribed antibiotic. "
            "Prevention: drink plenty of water, don't hold urine, urinate after intercourse."
        ),
        summary_mr=(
            "UTI मध्ये लघवी करताना जळजळ, वारंवार लघवी, खालच्या पोटात दुखणे, ढगाळ लघवी होते. "
            "संसर्ग मूत्रपिंडापर्यंत पोहोचल्यास ताप, थंडी, पाठदुखी, उलटी होते — त्वरित डॉक्टर गाठा. "
            "निदान: लघवीची तपासणी. उपचार: डॉक्टरांनी दिलेले प्रतिजैविक. "
            "प्रतिबंध: भरपूर पाणी प्या, लघवी रोखू नका."
        ),
        summary_hi=(
            "UTI में पेशाब करते समय जलन, बार-बार पेशाब, पेट के नीचे दर्द, बादली पेशाब होती है। "
            "संक्रमण किडनी तक पहुंचे तो बुखार, ठंड, पीठ दर्द, उल्टी — तुरंत डॉक्टर के पास जाएं। "
            "जांच: पेशाब की जांच (culture)। उपचार: डॉक्टर की दवा (antibiotic)। "
            "बचाव: खूब पानी पिएं, पेशाब रोकें नहीं।"
        ),
        source_name="NIH MedlinePlus — UTI",
        source_url="https://medlineplus.gov/urinarytractinfections.html",
    ),

    KBArticle(
        title_en="Stomach / Abdominal Pain",
        title_mr="पोटदुखी",
        title_hi="पेट दर्द",
        keywords=[
            "stomach pain", "belly pain", "abdominal pain", "abdomen", "stomach ache",
            "gastric", "acidity", "ulcer", "indigestion", "pet dard",
            "पोट दुखत", "पोट दुखणे", "पोट दुखतंय", "पेट दर्द", "पेट में दर्द", "पोट",
            "pot dukhtay", "pot dukhat", "poth dukhat", "pothala", "pothat", "potat",
            "dukhat", "dukhtay", "abdominal", "tummy", "belly",
        ],
        summary_en=(
            "Belly pain for weeks/months needs a doctor check. "
            "Common causes: acidity/gastritis, stomach ulcer, constipation, gallstones, "
            "infection (typhoid, worms), and in women, period or ovary problems. "
            "Kidney-related causes: kidney stones (pain in side spreading to groin), "
            "urine infection (burning urine + lower belly pain), or blocked urine flow. "
            "See a doctor soon for: weight loss, vomiting, black stools, blood in vomit/urine, fever, yellow eyes. "
            "Useful tests: blood tests, urine test, ultrasound."
        ),
        summary_mr=(
            "अनेक आठवडे किंवा महिने पोट दुखत असेल तर डॉक्टरांकडे जाणे आवश्यक आहे. "
            "सामान्य कारणे: ॲसिडिटी, जठरव्रण, बद्धकोष्ठता, पित्ताशयातील खडे, संसर्ग. "
            "मूत्रपिंडाशी संबंधित कारणे: किडनी स्टोन (कंबरेतून मांडीपर्यंत वेदना), लघवीचा संसर्ग. "
            "वजन कमी होणे, उलटी, काळी विष्ठा, रक्त असल्यास तातडीने डॉक्टरांकडे जा. "
            "तपासण्या: रक्त तपासणी, लघवी तपासणी, सोनोग्राफी."
        ),
        summary_hi=(
            "हफ्तों या महीनों से पेट दर्द हो तो डॉक्टर से जांच जरूरी है। "
            "आम कारण: एसिडिटी, अल्सर, कब्ज, पथरी (gallbladder), संक्रमण। "
            "किडनी से जुड़े कारण: किडनी स्टोन (बाजू से नीचे दर्द), पेशाब का संक्रमण। "
            "वजन कम होना, उल्टी, काला मल, खून हो तो तुरंत डॉक्टर के पास जाएं। "
            "जांच: खून, पेशाब, अल्ट्रासाउंड।"
        ),
        source_name="NIH MedlinePlus — Abdominal Pain",
        source_url="https://medlineplus.gov/abdominalpain.html",
    ),

    KBArticle(
        title_en="Diabetic Kidney Disease",
        title_mr="मधुमेहजन्य मूत्रपिंड आजार",
        title_hi="मधुमेह और किडनी",
        keywords=[
            "diabetes", "diabetic", "sugar", "dm", "blood sugar", "diabetic nephropathy",
            "मधुमेह", "शुगर", "साखर", "madhumeh", "sugar", "insulin",
        ],
        summary_en=(
            "High blood sugar slowly damages the kidney's tiny filters over years. "
            "First sign: protein (albumin) in urine (urine ACR test). Later: eGFR falls. "
            "Every diabetic should have yearly urine ACR and creatinine tests. "
            "Good sugar control (HbA1c target set by doctor) and BP control slow kidney damage significantly. "
            "Doctor-prescribed medicines can protect the kidney — never stop them without asking."
        ),
        summary_mr=(
            "जास्त रक्तातील साखरेमुळे वर्षानुवर्षे मूत्रपिंडातील लहान फिल्टर खराब होतात. "
            "पहिले लक्षण: लघवीत प्रथिने (albumin) — ACR तपासणीने दिसते. नंतर eGFR कमी होते. "
            "प्रत्येक मधुमेही रुग्णाने दरवर्षी ACR आणि creatinine तपासणी करावी. "
            "रक्तातील साखर (HbA1c) आणि रक्तदाब नियंत्रणात ठेवल्यास मूत्रपिंड जास्त काळ टिकते. "
            "डॉक्टरांनी दिलेली औषधे मूत्रपिंडाचे रक्षण करतात — स्वतःहून बंद करू नका."
        ),
        summary_hi=(
            "ज्यादा रक्त शर्करा से सालों में किडनी के छोटे फिल्टर खराब होते हैं। "
            "पहला संकेत: पेशाब में प्रोटीन (ACR जांच)। बाद में eGFR घटता है। "
            "हर मधुमेही को हर साल ACR और creatinine की जांच करानी चाहिए। "
            "HbA1c और BP नियंत्रण से किडनी का नुकसान काफी कम होता है। "
            "डॉक्टर की दवाएं किडनी की रक्षा करती हैं — बिना पूछे बंद न करें।"
        ),
        source_name="KDIGO 2022 — Diabetes Management in CKD",
        source_url="https://kdigo.org/guidelines/diabetes-ckd/",
    ),

    KBArticle(
        title_en="High Blood Pressure & Kidneys",
        title_mr="उच्च रक्तदाब आणि मूत्रपिंड",
        title_hi="उच्च रक्तचाप और किडनी",
        keywords=[
            "high blood pressure", "hypertension", "bp", "blood pressure", "htn",
            "रक्तदाब", "ब्लड प्रेशर", "blood pressure high", "bp high",
            "raktadab", "raktachaap",
        ],
        summary_en=(
            "Long-standing high BP slowly damages kidney blood vessels, and damaged kidneys "
            "raise BP further — a worsening cycle. "
            "Target BP for people with kidney disease is usually below 130/80 (doctor decides). "
            "Take BP medicines every day without missing doses. "
            "Lifestyle: cut salt to under 5g/day, exercise regularly, maintain healthy weight, stop smoking."
        ),
        summary_mr=(
            "दीर्घकाळ उच्च रक्तदाबामुळे मूत्रपिंडातील रक्तवाहिन्या खराब होतात, "
            "आणि खराब मूत्रपिंड रक्तदाब आणखी वाढवतात — हे चक्र वाईट होत जाते. "
            "मूत्रपिंड आजार असलेल्यांसाठी रक्तदाब १३०/८० पेक्षा कमी असावा (डॉक्टर ठरवतात). "
            "रक्तदाबाची औषधे रोज न चुकता घ्या. "
            "आहार: मीठ ५ ग्रॅमपेक्षा कमी, व्यायाम, वजन नियंत्रण, धूम्रपान बंद."
        ),
        summary_hi=(
            "लंबे समय तक उच्च BP से किडनी की रक्त नलिकाएं खराब होती हैं, "
            "और खराब किडनी BP और बढ़ाती है — यह एक दुष्चक्र है। "
            "किडनी रोगियों का BP लक्ष्य आमतौर पर 130/80 से कम (डॉक्टर तय करते हैं)। "
            "BP की दवाएं रोज बिना नागा लें। "
            "जीवनशैली: नमक कम, रोज व्यायाम, वजन नियंत्रण, धूम्रपान बंद।"
        ),
        source_name="NKF — High Blood Pressure & CKD",
        source_url="https://www.kidney.org/kidney-topics/high-blood-pressure-hypertension",
    ),

    KBArticle(
        title_en="Dialysis & Kidney Transplant",
        title_mr="डायलिसिस आणि किडनी प्रत्यारोपण",
        title_hi="डायलिसिस और किडनी ट्रांसप्लांट",
        keywords=[
            "dialysis", "haemodialysis", "hemodialysis", "transplant", "peritoneal",
            "fistula", "kidney replacement", "डायलिसिस", "प्रत्यारोपण", "kidney transplant",
        ],
        summary_en=(
            "Dialysis is needed when kidneys fail (eGFR usually below 10–15 with symptoms). "
            "Haemodialysis: blood cleaned by machine 3 times/week for ~4 hours each; requires a fistula. "
            "Peritoneal dialysis: uses the belly lining; can be done at home. "
            "Kidney transplant gives the best quality of life; donors can be living or deceased. "
            "Government schemes (PM-JAY, state schemes) may cover dialysis costs — ask a hospital social worker."
        ),
        summary_mr=(
            "मूत्रपिंड निकामी झाल्यावर (eGFR साधारण १०–१५ पेक्षा कमी) डायलिसिस लागते. "
            "हिमोडायलिसिस: आठवड्यात ३ वेळा ~४ तास मशीनने रक्त स्वच्छ करणे; फिस्टुला लागतो. "
            "पेरिटोनियल डायलिसिस: पोटाचा पडदा वापरतो; घरी करता येते. "
            "किडनी प्रत्यारोपण सर्वात चांगले जीवन देते; दाता जिवंत किंवा मृत असू शकतो. "
            "PM-JAY सारख्या सरकारी योजनांमधून डायलिसिसचा खर्च मिळू शकतो — रुग्णालयाच्या समाज कार्यकर्त्याला विचारा."
        ),
        summary_hi=(
            "किडनी फेल होने पर (eGFR आमतौर पर 10–15 से कम) डायलिसिस जरूरी होता है। "
            "हेमोडायलिसिस: हफ्ते में 3 बार ~4 घंटे मशीन से खून साफ होता है; fistula चाहिए। "
            "पेरिटोनियल डायलिसिस: पेट की परत का उपयोग; घर पर हो सकता है। "
            "किडनी ट्रांसप्लांट सबसे अच्छी जिंदगी देता है; दाता जीवित या मृत हो सकता है। "
            "PM-JAY जैसी सरकारी योजनाओं से डायलिसिस का खर्च मिल सकता है।"
        ),
        source_name="NKF — Dialysis",
        source_url="https://www.kidney.org/kidney-topics/dialysis",
    ),

    KBArticle(
        title_en="Kidney Diet & Lifestyle",
        title_mr="मूत्रपिंड आजारात आहार आणि जीवनशैली",
        title_hi="किडनी रोग में आहार और जीवनशैली",
        keywords=[
            "diet", "food", "eat", "salt", "potassium", "protein", "water", "drink",
            "banana", "phosphorus", "आहार", "खाना", "खाना पीना", "aahar", "jevan",
            "potassium food", "banana kidney",
        ],
        summary_en=(
            "General kidney health: less than 5g salt/day, 2–2.5L water (unless restricted by doctor), "
            "more vegetables, less processed food and sugary drinks, regular exercise, no smoking. "
            "In CKD (advanced stages): limit potassium (banana, coconut water, tomato, potato), "
            "phosphorus (colas, processed food), and sometimes protein and fluids. "
            "Diets in CKD are personalised — always consult a renal dietitian before making changes."
        ),
        summary_mr=(
            "मूत्रपिंड आरोग्यासाठी: दिवसातून ५ ग्रॅमपेक्षा कमी मीठ, २–२.५ लिटर पाणी "
            "(डॉक्टरांनी नाही म्हटल्यास), जास्त भाज्या, कमी प्रक्रिया केलेले खाद्यपदार्थ, व्यायाम, धूम्रपान बंद. "
            "CKD च्या प्रगत अवस्थेत: पोटॅशियम (केळी, नारळ पाणी, टमाटे), फॉस्फरस (कोला) कमी करा. "
            "CKD मधील आहार प्रत्येकासाठी वेगळा असतो — तज्ज्ञ आहारतज्ज्ञाचा सल्ला घ्या."
        ),
        summary_hi=(
            "किडनी स्वास्थ्य के लिए: दिन में 5 ग्राम से कम नमक, 2–2.5 लीटर पानी "
            "(डॉक्टर ने मना न किया हो), ज्यादा सब्जियां, कम प्रोसेस्ड खाना, व्यायाम, धूम्रपान बंद। "
            "CKD में: पोटैशियम (केला, नारियल पानी, टमाटर) और फॉस्फोरस (कोला) कम करें। "
            "CKD की डाइट सबके लिए अलग होती है — renal dietitian से सलाह लें।"
        ),
        source_name="NKF — Kidney Diet Basics",
        source_url="https://www.kidney.org/kidney-topics/nutrition-kidney-disease",
    ),

    KBArticle(
        title_en="Kidney Function Tests — Creatinine, eGFR, Urine ACR",
        title_mr="मूत्रपिंड तपासण्या — Creatinine, eGFR, ACR",
        title_hi="किडनी जांच — Creatinine, eGFR, ACR",
        keywords=[
            "creatinine", "egfr", "acr", "urine protein", "urine albumin",
            "blood test", "kidney test", "kidney report", "urea", "bun",
            "serum creatinine", "creatinine level", "eGFR low",
            "creatinine", "report", "thapaasni", "tapaasni",
        ],
        summary_en=(
            "Serum creatinine (normal ~0.6–1.2 mg/dL; lower in women/young) is used with age and sex "
            "to calculate eGFR (normal ≥90 mL/min/1.73m²). "
            "Urine ACR (albumin-to-creatinine ratio): <30 mg/g normal; 30–300 mildly raised; >300 markedly raised. "
            "Blood urea (BUN): normal ~15–40 mg/dL (higher with dehydration or high-protein diet). "
            "A single abnormal result should be repeated — CKD requires 3+ months of abnormality."
        ),
        summary_mr=(
            "Serum creatinine (सामान्य ~0.6–1.2 mg/dL) वय आणि लिंगासह eGFR काढण्यासाठी वापरतात. "
            "eGFR सामान्य: ≥90 mL/min/1.73m². "
            "Urine ACR (albumin-creatinine): <30 mg/g सामान्य; 30–300 थोडे जास्त; >300 खूप जास्त. "
            "Blood urea (BUN): सामान्य ~15–40 mg/dL. "
            "एक वेळचा असामान्य निकाल परत तपासावा — CKD साठी ३+ महिने असामान्य असणे आवश्यक."
        ),
        summary_hi=(
            "Serum creatinine (सामान्य ~0.6–1.2 mg/dL) उम्र और लिंग के साथ eGFR निकालने में काम आता है। "
            "eGFR सामान्य: ≥90 mL/min/1.73m²। "
            "Urine ACR: <30 mg/g सामान्य; 30–300 थोड़ा अधिक; >300 बहुत अधिक। "
            "BUN सामान्य ~15–40 mg/dL। "
            "एक बार की असामान्य जांच को दोबारा कराएं — CKD के लिए 3+ महीने चाहिए।"
        ),
        source_name="KDIGO — CKD Evaluation",
        source_url="https://kdigo.org/guidelines/ckd-evaluation-and-management/",
    ),

    KBArticle(
        title_en="NSAIDs & Medicines That Harm Kidneys",
        title_mr="वेदनाशामक आणि मूत्रपिंडाला हानी करणारी औषधे",
        title_hi="दर्दनिवारक और किडनी को नुकसान पहुंचाने वाली दवाएं",
        keywords=[
            "painkiller", "nsaid", "ibuprofen", "diclofenac", "brufen", "voveran",
            "medicine", "herbal", "ayurvedic", "contrast dye", "painkillers kidney",
            "vedanashamak", "dard ki dawa",
        ],
        summary_en=(
            "NSAID painkillers (ibuprofen, diclofenac, naproxen) can injure kidneys, especially with "
            "dehydration, diabetes, high BP, or existing CKD. Even occasional use adds risk. "
            "Some herbal/ayurvedic preparations with heavy metals also harm kidneys. "
            "Always tell your doctor about kidney problems before any new medicine or contrast CT scan. "
            "Never stop prescribed medicines on your own — ask your doctor first."
        ),
        summary_mr=(
            "NSAID वेदनाशामक (ibuprofen, diclofenac) मूत्रपिंडाला हानी करतात, विशेषतः पाण्याची कमतरता, "
            "मधुमेह किंवा आधीच CKD असताना. "
            "काही हर्बल/आयुर्वेदिक तयारींमध्ये जड धातू असतात जे मूत्रपिंडाला हानी करतात. "
            "कोणतेही नवीन औषध किंवा contrast CT scan आधी डॉक्टरांना मूत्रपिंड आजाराबद्दल सांगा. "
            "स्वतःहून कोणतेही औषध बंद करू नका — आधी डॉक्टरांना विचारा."
        ),
        summary_hi=(
            "NSAID दर्दनिवारक (ibuprofen, diclofenac) खासकर पानी की कमी, मधुमेह या CKD के साथ किडनी को नुकसान पहुंचाते हैं। "
            "कुछ हर्बल/आयुर्वेदिक दवाओं में भारी धातुएं होती हैं जो किडनी को नुकसान पहुंचाती हैं। "
            "कोई भी नई दवा या contrast CT scan से पहले डॉक्टर को किडनी की बीमारी बताएं। "
            "खुद से कोई दवा बंद न करें — पहले डॉक्टर से पूछें।"
        ),
        source_name="NKF — Medications & CKD",
        source_url="https://www.kidney.org/kidney-topics/pain-medicines-nsaids-and-your-kidneys",
    ),

    KBArticle(
        title_en="Anaemia in Kidney Disease",
        title_mr="मूत्रपिंड आजारात अशक्तपणा (Anaemia)",
        title_hi="किडनी रोग में एनीमिया",
        keywords=[
            "anaemia", "anemia", "haemoglobin", "hemoglobin", "hb low", "tired",
            "fatigue", "breathless", "pale", "weakness",
            "अशक्तपणा", "हिमोग्लोबिन", "थकवा", "anemia", "khoon ki kami",
        ],
        summary_en=(
            "Damaged kidneys produce less erythropoietin (EPO), causing anaemia — low haemoglobin "
            "→ tiredness, pale skin, breathlessness, rapid heartbeat. "
            "Checked with a CBC blood test. Treated with iron supplements, EPO injections, or both. "
            "Anaemia in CKD is treatable — tell your nephrologist if you feel very tired or breathless."
        ),
        summary_mr=(
            "खराब मूत्रपिंड कमी erythropoietin (EPO) तयार करते, ज्यामुळे अशक्तपणा होतो — "
            "कमी hemoglobin → थकवा, फिका रंग, धाप लागणे, हृदयाचे ठोके जलद होणे. "
            "CBC रक्त तपासणीने आढळते. लोह पूरक, EPO इंजेक्शन किंवा दोन्ही उपचार आहेत. "
            "CKD मधील अशक्तपणा उपचारयोग्य आहे — खूप थकवा किंवा धाप असल्यास nephrologist ला सांगा."
        ),
        summary_hi=(
            "खराब किडनी कम erythropoietin (EPO) बनाती है, जिससे एनीमिया होता है — "
            "कम hemoglobin → थकान, पीलापन, सांस फूलना, दिल की धड़कन तेज। "
            "CBC रक्त जांच से पता चलता है। आयरन सप्लीमेंट, EPO इंजेक्शन या दोनों से इलाज होता है। "
            "CKD का एनीमिया इलाज योग्य है — बहुत थकान या सांस फूलने पर nephrologist को बताएं।"
        ),
        source_name="KDIGO 2024 — Anaemia in CKD",
        source_url="https://kdigo.org/guidelines/anaemia-in-ckd/",
    ),
]

# Pre-build a flat keyword index for fast lookup
_INDEX: list[tuple[set[str], KBArticle]] = [
    (set(a.keywords), a) for a in ARTICLES
]


def retrieve_context(query: str, lang: str, k: int = 2) -> list[dict]:
    """
    Score each article by keyword overlap with the query.
    Returns up to k articles as dicts with keys:
      title, summary, source_name, source_url
    lang: 'english' | 'hindi' | 'marathi'
    """
    q_tokens = set(re.findall(r"[\w\u0900-\u097F]+", (query or "").lower()))

    scored: list[tuple[float, KBArticle]] = []
    for kw_set, article in _INDEX:
        # exact token matches
        exact = len(q_tokens & kw_set)
        # prefix matches (e.g. "kidney" matches "kidney stone")
        prefix = sum(
            1 for qt in q_tokens
            for kw in kw_set
            if len(qt) >= 4 and kw.startswith(qt)
        )
        score = exact + 0.5 * prefix
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for _, article in scored[:k]:
        summary = {
            "english": article.summary_en,
            "hindi":   article.summary_hi,
            "marathi": article.summary_mr,
        }.get(lang, article.summary_en)

        title = {
            "english": article.title_en,
            "hindi":   article.title_hi,
            "marathi": article.title_mr,
        }.get(lang, article.title_en)

        results.append({
            "title":       title,
            "summary":     summary,
            "source_name": article.source_name,
            "source_url":  article.source_url,
        })
    return results


def format_context_for_prompt(articles: list[dict], lang: str) -> str:
    """Format retrieved articles into a prompt section."""
    if not articles:
        return ""
    header = {
        "english": "VERIFIED MEDICAL INFORMATION (use this to answer, include source):",
        "hindi":   "प्रमाणित चिकित्सा जानकारी (इसका उपयोग करके उत्तर दें, स्रोत बताएं):",
        "marathi": "प्रमाणित वैद्यकीय माहिती (याचा वापर करून उत्तर द्या, स्रोत सांगा):",
    }.get(lang, "VERIFIED MEDICAL INFORMATION:")

    parts = [header]
    for a in articles:
        parts.append(f"\n[{a['title']}]\n{a['summary']}\nSource: {a['source_name']} — {a['source_url']}")
    return "\n".join(parts)
