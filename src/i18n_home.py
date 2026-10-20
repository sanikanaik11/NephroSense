"""
i18n_home.py — NephroSense
Translations for the test-free Home Screening page.
Supports English (en), Hindi (hi), Marathi (mr).
"""
from __future__ import annotations

LANGS = {"en": "English", "hi": "हिंदी", "mr": "मराठी"}

_T: dict[str, dict[str, str]] = {
    # Page header
    "title": {
        "en": "Home CKD Risk Screen",
        "hi": "घर पर CKD जोखिम जांच",
        "mr": "घरगुती CKD धोका तपासणी",
    },
    "subtitle": {
        "en": "No blood or urine test needed — answer questions about your health",
        "hi": "खून या पेशाब की जांच जरूरी नहीं — अपने स्वास्थ्य के बारे में जवाब दें",
        "mr": "रक्त किंवा लघवी तपासणी लागत नाही — तुमच्या आरोग्याबद्दल प्रश्नांची उत्तरे द्या",
    },
    "notice": {
        "en": "⚠️ This is a screening tool, not a diagnosis. Always confirm results with a doctor.",
        "hi": "⚠️ यह एक स्क्रीनिंग टूल है, निदान नहीं। परिणामों की पुष्टि हमेशा डॉक्टर से करें।",
        "mr": "⚠️ हे एक स्क्रीनिंग साधन आहे, निदान नाही. परिणाम नेहमी डॉक्टरांकडून तपासा.",
    },
    # Consent
    "consent_title": {"en": "Before you start", "hi": "शुरू करने से पहले", "mr": "सुरू करण्यापूर्वी"},
    "consent_body": {
        "en": "This tool gives an estimate of your CKD risk based on your answers. It cannot diagnose kidney disease. Your data is stored securely and only you can see it.",
        "hi": "यह टूल आपके जवाबों के आधार पर CKD जोखिम का अनुमान देता है। यह किडनी रोग का निदान नहीं कर सकता। आपका डेटा सुरक्षित रखा जाता है।",
        "mr": "हे साधन तुमच्या उत्तरांवर आधारित CKD धोक्याचा अंदाज देते. हे मूत्रपिंड आजाराचे निदान करू शकत नाही. तुमचा डेटा सुरक्षित ठेवला जातो.",
    },
    "consent_check": {
        "en": "I understand this is a screening tool, not a medical diagnosis.",
        "hi": "मैं समझता/समझती हूं कि यह एक स्क्रीनिंग टूल है, चिकित्सा निदान नहीं।",
        "mr": "मला समजते की हे एक स्क्रीनिंग साधन आहे, वैद्यकीय निदान नाही.",
    },
    "consent_btn": {"en": "I agree — continue", "hi": "मैं सहमत हूं — जारी रखें", "mr": "मी सहमत आहे — पुढे चला"},
    # Health-worker toggle
    "hw_toggle": {"en": "I am a health worker screening a patient", "hi": "मैं एक स्वास्थ्य कार्यकर्ता हूं", "mr": "मी एक आरोग्य कार्यकर्ता आहे"},
    "hw_code": {"en": "Patient code (letters & numbers only)", "hi": "रोगी कोड", "mr": "रुग्ण कोड"},
    "hw_code_bad": {"en": "Enter a code of 3–20 letters or digits.", "hi": "3–20 अक्षर या अंक का कोड दर्ज करें।", "mr": "3–20 अक्षरे किंवा अंकांचा कोड द्या."},
    # Step 1 — symptoms
    "step1": {"en": "Step 1 — Warning signs", "hi": "चरण 1 — चेतावनी के लक्षण", "mr": "पायरी 1 — धोक्याची लक्षणे"},
    "step1_help": {
        "en": "Tick any you have RIGHT NOW. If you tick a red warning, the page will tell you to go to hospital.",
        "hi": "अभी जो भी हैं उन पर टिक करें। लाल चेतावनी टिक करने पर अस्पताल जाने का संदेश आएगा।",
        "mr": "आत्ता जे आहे ते टिक करा. लाल इशारा टिक केल्यास रुग्णालयात जाण्याचा संदेश येईल.",
    },
    # Flag labels
    "f_little_urine": {"en": "Very little or no urine output", "hi": "बहुत कम या बिल्कुल पेशाब नहीं", "mr": "खूप कमी किंवा अजिबात लघवी नाही"},
    "f_breathless":   {"en": "Breathless at rest or lying down", "hi": "आराम में या लेटने पर सांस फूलना", "mr": "विश्रांती किंवा झोपताना श्वास लागणे"},
    "f_chest_pain":   {"en": "Chest pain or pressure", "hi": "छाती में दर्द या दबाव", "mr": "छातीत दुखणे किंवा दाब"},
    "f_confusion":    {"en": "Confusion, fits or unusual drowsiness", "hi": "भ्रम, दौरे या असामान्य नींद", "mr": "गोंधळ, झटके किंवा असामान्य झोप"},
    "f_swelling":     {"en": "Swelling of feet, ankles or face", "hi": "पैर, टखने या चेहरे में सूजन", "mr": "पाय, घोटे किंवा चेहरा सुजणे"},
    "f_vomiting":     {"en": "Repeated vomiting", "hi": "बार-बार उल्टी", "mr": "वारंवार उलट्या"},
    "f_blood_urine":  {"en": "Blood or cola-coloured urine", "hi": "खून या कोला रंग का पेशाब", "mr": "रक्त किंवा कोला रंगाची लघवी"},
    "f_foamy":        {"en": "Foamy urine", "hi": "झागदार पेशाब", "mr": "फेसाळ लघवी"},
    "f_more_urine":   {"en": "Much more frequent urination", "hi": "बहुत बार पेशाब आना", "mr": "खूप वारंवार लघवी"},
    "f_night_urine":  {"en": "Waking 2+ times at night to urinate", "hi": "रात में 2+ बार पेशाब के लिए उठना", "mr": "रात्री 2+ वेळा लघवीसाठी उठणे"},
    "f_tired":        {"en": "Unusual tiredness or weakness", "hi": "असामान्य थकान या कमजोरी", "mr": "असामान्य थकवा किंवा अशक्तपणा"},
    "f_poor_appetite":{"en": "Poor appetite or unexplained weight loss", "hi": "भूख न लगना या वजन कम होना", "mr": "भूक न लागणे किंवा वजन कमी होणे"},
    "f_nausea":       {"en": "Nausea", "hi": "मतली", "mr": "मळमळ"},
    "f_itch":         {"en": "Itching of the skin all over", "hi": "पूरे शरीर में खुजली", "mr": "संपूर्ण अंगावर खाज"},
    "f_flank":        {"en": "Pain in sides or lower back", "hi": "बाजू या पीठ के निचले हिस्से में दर्द", "mr": "कंबर किंवा पाठीच्या खालच्या भागात दुखणे"},
    "f_burning":      {"en": "Burning while urinating", "hi": "पेशाब करते समय जलन", "mr": "लघवी करताना जळजळ"},
    # Emergency / urgent messages
    "emergency_title": {"en": "Go to hospital immediately", "hi": "तुरंत अस्पताल जाएं", "mr": "तातडीने रुग्णालयात जा"},
    "emergency_msg": {
        "en": "You have ticked a warning sign that needs urgent hospital care. Please go to the nearest hospital or call 112 now. Do not wait.",
        "hi": "आपने एक चेतावनी संकेत टिक किया है जिसे तुरंत अस्पताल देखभाल की आवश्यकता है। अभी निकटतम अस्पताल जाएं या 112 पर कॉल करें।",
        "mr": "तुम्ही एक धोकादायक लक्षण टिक केले आहे ज्यासाठी तातडीने रुग्णालयात जाणे आवश्यक आहे. आत्ता जवळच्या रुग्णालयात जा किंवा 112 वर कॉल करा.",
    },
    "urgent_title": {"en": "See a doctor today", "hi": "आज डॉक्टर से मिलें", "mr": "आज डॉक्टरांकडे जा"},
    "urgent_msg": {
        "en": "You have symptoms that need a doctor's assessment today — do not wait.",
        "hi": "आपके लक्षण आज डॉक्टर की जांच की मांग करते हैं — प्रतीक्षा न करें।",
        "mr": "तुमची लक्षणे आज डॉक्टरांच्या तपासणीची मागणी करतात — थांबू नका.",
    },
    "find_btn": {"en": "Find a hospital nearby →", "hi": "पास का अस्पताल खोजें →", "mr": "जवळचे रुग्णालय शोधा →"},
    # Step 2 — BP
    "step2": {"en": "Step 2 — 7-day home blood pressure (optional but helpful)", "hi": "चरण 2 — 7-दिन का घर पर BP (वैकल्पिक)", "mr": "पायरी 2 — 7-दिवसांचा घरगुती BP (पर्यायी)"},
    "bp_howto": {
        "en": "Measure your BP in the morning and evening for 7 days. Enter each reading below. We discard the very first reading (day 1, morning) as people are often nervous.",
        "hi": "7 दिनों तक सुबह और शाम BP मापें। हर पठन नीचे दर्ज करें।",
        "mr": "7 दिवस सकाळी आणि संध्याकाळी BP मोजा. प्रत्येक वाचन खाली नोंदवा.",
    },
    "bp_day":      {"en": "Day (1–7)", "hi": "दिन (1–7)", "mr": "दिवस (1–7)"},
    "bp_period":   {"en": "When", "hi": "कब", "mr": "केव्हा"},
    "bp_am":       {"en": "Morning", "hi": "सुबह", "mr": "सकाळ"},
    "bp_pm":       {"en": "Evening", "hi": "शाम", "mr": "संध्याकाळ"},
    "bp_sys":      {"en": "Upper (SBP)", "hi": "ऊपरी (SBP)", "mr": "वरील (SBP)"},
    "bp_dia":      {"en": "Lower (DBP)", "hi": "निचला (DBP)", "mr": "खालील (DBP)"},
    "bp_add":      {"en": "Add reading", "hi": "पठन जोड़ें", "mr": "वाचन जोडा"},
    "bp_clear":    {"en": "Clear all", "hi": "सब हटाएं", "mr": "सर्व हटवा"},
    "bp_bad":      {"en": "Invalid BP reading — upper must be larger than lower, and both must be realistic.", "hi": "अमान्य BP — ऊपरी निचले से अधिक होना चाहिए।", "mr": "अवैध BP — वरील खालीलपेक्षा जास्त असणे आवश्यक."},
    "bp_progress": {"en": "{n} of {need} readings entered", "hi": "{n}/{need} पठन दर्ज", "mr": "{n}/{need} वाचने नोंदवली"},
    "bp_avg":      {"en": "Average BP: {s}/{d} mmHg", "hi": "औसत BP: {s}/{d} mmHg", "mr": "सरासरी BP: {s}/{d} mmHg"},
    "bp_raised":   {"en": "This is above the normal range — please discuss with a doctor.", "hi": "यह सामान्य सीमा से ऊपर है — डॉक्टर से बात करें।", "mr": "हे सामान्य श्रेणीपेक्षा जास्त आहे — डॉक्टरांशी बोला."},
    "bp_normal":   {"en": "This is within the normal range.", "hi": "यह सामान्य सीमा में है।", "mr": "हे सामान्य श्रेणीत आहे."},
    # Step 3 — questions
    "step3":       {"en": "Step 3 — Your health", "hi": "चरण 3 — आपका स्वास्थ्य", "mr": "पायरी 3 — तुमचे आरोग्य"},
    "choose":      {"en": "— choose —", "hi": "— चुनें —", "mr": "— निवडा —"},
    "yes":         {"en": "Yes", "hi": "हाँ", "mr": "हो"},
    "no":          {"en": "No", "hi": "नहीं", "mr": "नाही"},
    "female":      {"en": "Female", "hi": "महिला", "mr": "स्त्री"},
    "male":        {"en": "Male", "hi": "पुरुष", "mr": "पुरुष"},
    "l_age":       {"en": "Age (years)", "hi": "उम्र (वर्ष)", "mr": "वय (वर्षे)"},
    "l_sex":       {"en": "Sex", "hi": "लिंग", "mr": "लिंग"},
    "l_height":    {"en": "Height (cm)", "hi": "ऊंचाई (सेमी)", "mr": "उंची (सेमी)"},
    "l_weight":    {"en": "Weight (kg)", "hi": "वजन (किलो)", "mr": "वजन (किलो)"},
    "l_waist":     {"en": "Waist at navel (cm) — optional", "hi": "नाभि पर कमर (सेमी) — वैकल्पिक", "mr": "नाभीवर कंबर (सेमी) — पर्यायी"},
    "bmi_line":    {"en": "BMI: {v} kg/m²", "hi": "BMI: {v} kg/m²", "mr": "BMI: {v} kg/m²"},
    "l_diabetes":  {"en": "Diabetes (told by a doctor)?", "hi": "मधुमेह (डॉक्टर ने बताया)?", "mr": "मधुमेह (डॉक्टरांनी सांगितले)?"},
    "l_htn_meds":  {"en": "Taking blood pressure medicine?", "hi": "BP की दवा ले रहे हैं?", "mr": "BP चे औषध घेत आहात?"},
    "l_heart":     {"en": "Heart attack, angina or heart failure?", "hi": "दिल का दौरा, एनजाइना या दिल की विफलता?", "mr": "हृदयविकाराचा झटका, एनजायना किंवा हृदय अपयश?"},
    "l_stones":    {"en": "Kidney stones (ever)?", "hi": "कभी गुर्दे की पथरी?", "mr": "कधी किडनी स्टोन?"},
    "extras_title":{"en": "Two more questions", "hi": "दो और प्रश्न", "mr": "आणखी दोन प्रश्न"},
    "l_nsaid":     {"en": "Take ibuprofen / diclofenac painkillers regularly?", "hi": "नियमित ibuprofen/diclofenac दर्दनिवारक लेते हैं?", "mr": "नियमितपणे ibuprofen/diclofenac वेदनाशामक घेता?"},
    "l_family":    {"en": "Parent or sibling with kidney disease or dialysis?", "hi": "माता-पिता या भाई-बहन को किडनी रोग?", "mr": "आई-वडील किंवा भाऊ-बहीण यांना किडनी आजार?"},
    "l_strip":     {"en": "Urine dipstick protein — if you have one (optional)", "hi": "यूरिन डिपस्टिक प्रोटीन — अगर है (वैकल्पिक)", "mr": "लघवी डिपस्टिक प्रोटीन — असल्यास (पर्यायी)"},
    "s_none":      {"en": "Not done / don't know", "hi": "नहीं किया / नहीं पता", "mr": "केले नाही / माहित नाही"},
    "s_negative":  {"en": "Negative (no protein)", "hi": "नकारात्मक (कोई प्रोटीन नहीं)", "mr": "नकारात्मक (प्रोटीन नाही)"},
    "s_trace":     {"en": "Trace", "hi": "ट्रेस", "mr": "ट्रेस"},
    "s_30":        {"en": "1+ (30 mg/dL)", "hi": "1+ (30 mg/dL)", "mr": "1+ (30 mg/dL)"},
    "s_100":       {"en": "2+ (100 mg/dL)", "hi": "2+ (100 mg/dL)", "mr": "2+ (100 mg/dL)"},
    "s_300":       {"en": "3+ / 4+ (300+ mg/dL)", "hi": "3+/4+ (300+ mg/dL)", "mr": "3+/4+ (300+ mg/dL)"},
    "check_btn":   {"en": "Check my risk →", "hi": "मेरा जोखिम जांचें →", "mr": "माझा धोका तपासा →"},
    # Result
    "res_title":   {"en": "Your result", "hi": "आपका परिणाम", "mr": "तुमचा निकाल"},
    "no_model":    {"en": "The risk model is not available on this computer. Please run: python src/home_risk.py --train", "hi": "जोखिम मॉडल उपलब्ध नहीं है।", "mr": "धोका मॉडेल उपलब्ध नाही."},
    "band_Low":    {"en": "Low risk", "hi": "कम जोखिम", "mr": "कमी धोका"},
    "band_Possible":{"en":"Possible risk", "hi": "संभावित जोखिम", "mr": "संभाव्य धोका"},
    "band_High":   {"en": "Higher risk", "hi": "अधिक जोखिम", "mr": "अधिक धोका"},
    "advice_Low":  {"en": "Your answers suggest a lower likelihood of CKD at this time. Repeat this screen once a year, especially if you have diabetes or high BP.", "hi": "आपके जवाब अभी CKD की कम संभावना दर्शाते हैं। साल में एक बार दोहराएं।", "mr": "तुमच्या उत्तरांनुसार सध्या CKD ची शक्यता कमी आहे. दरवर्षी पुन्हा करा."},
    "advice_Possible":{"en":"Some answers suggest possible kidney risk factors. A creatinine and urine protein test is recommended.", "hi": "कुछ जवाब संभावित किडनी जोखिम कारक दर्शाते हैं। क्रिएटिनिन और यूरिन प्रोटीन परीक्षण की सिफारिश है।", "mr": "काही उत्तरे संभाव्य किडनी धोका घटक दर्शवतात. क्रिएटिनिन आणि लघवी प्रोटीन तपासणी शिफारसीय आहे."},
    "advice_High": {"en": "Your answers suggest a higher likelihood of kidney risk. Please see a doctor soon and ask for a creatinine and urine protein test.", "hi": "आपके जवाब किडनी जोखिम की उच्च संभावना दर्शाते हैं। जल्द डॉक्टर से मिलें।", "mr": "तुमच्या उत्तरांनुसार किडनी धोक्याची शक्यता जास्त आहे. लवकरात लवकर डॉक्टरांकडे जा."},
    "why_up":      {"en": "Factors raising your risk:", "hi": "जोखिम बढ़ाने वाले कारण:", "mr": "धोका वाढवणारे घटक:"},
    "why_down":    {"en": "Factors lowering your risk:", "hi": "जोखिम कम करने वाले कारण:", "mr": "धोका कमी करणारे घटक:"},
    "invalid_title":{"en":"Some values look wrong", "hi": "कुछ मान गलत लगते हैं", "mr": "काही मूल्ये चुकीची वाटतात"},
    "invalid_body":{"en": "Please check: {items}", "hi": "कृपया जाँचें: {items}", "mr": "कृपया तपासा: {items}"},
    "age_range":   {"en": "Age must be 18–100, and all measurements must be realistic.", "hi": "उम्र 18–100 होनी चाहिए।", "mr": "वय 18–100 असणे आवश्यक."},
    "refuse_title":{"en": "Not enough information", "hi": "पर्याप्त जानकारी नहीं", "mr": "पुरेशी माहिती नाही"},
    "refuse_body": {"en": "The following are needed for a result: {items}", "hi": "परिणाम के लिए ये जरूरी हैं: {items}", "mr": "निकालासाठी हे आवश्यक आहे: {items}"},
    "rel_title":   {"en": "About this result", "hi": "इस परिणाम के बारे में", "mr": "या निकालाबद्दल"},
    "rel_body":    {"en": "This model was trained on {n} records (cross-validated AUC {auc}, sensitivity {sens}%, specificity {spec}% at the Low/Possible threshold). It is experimental — confirm with a doctor.", "hi": "यह मॉडल {n} रिकॉर्ड पर प्रशिक्षित था।", "mr": "हे मॉडेल {n} नोंदींवर प्रशिक्षित होते."},
    "next_title":  {"en": "What to do next", "hi": "अगला कदम", "mr": "पुढे काय करावे"},
    "next_tests":  {"en": "Ask your doctor for: serum creatinine + eGFR, urine albumin-creatinine ratio (ACR), blood glucose (HbA1c if diabetic), BP check.", "hi": "डॉक्टर से मांगें: सीरम क्रिएटिनिन + eGFR, यूरिन ACR, ब्लड ग्लूकोज, BP जांच।", "mr": "डॉक्टरांकडून मागवा: सीरम क्रिएटिनिन + eGFR, लघवी ACR, रक्त ग्लुकोज, BP तपासणी."},
    "eye_note":    {"en": "You can also try the experimental eye screening (anemia hint only — not a blood test).", "hi": "आप प्रायोगिक नेत्र स्क्रीनिंग भी आजमा सकते हैं।", "mr": "तुम्ही प्रायोगिक डोळा स्क्रीनिंग देखील करून पाहू शकता."},
    "eye_btn":     {"en": "Try Eye Screening →", "hi": "नेत्र स्क्रीनिंग आज़माएं →", "mr": "डोळा स्क्रीनिंग करून पहा →"},
    # Feature labels (for result explanation)
    "ft_age":      {"en": "Age", "hi": "उम्र", "mr": "वय"},
    "ft_female":   {"en": "Sex", "hi": "लिंग", "mr": "लिंग"},
    "ft_sbp":      {"en": "Systolic BP", "hi": "सिस्टोलिक BP", "mr": "सिस्टोलिक BP"},
    "ft_dbp":      {"en": "Diastolic BP", "hi": "डायस्टोलिक BP", "mr": "डायस्टोलिक BP"},
    "ft_htn_meds": {"en": "On BP medicine", "hi": "BP दवा पर", "mr": "BP औषधावर"},
    "ft_diabetes":  {"en": "Diabetes", "hi": "मधुमेह", "mr": "मधुमेह"},
    "ft_bmi":       {"en": "BMI", "hi": "BMI", "mr": "BMI"},
    "ft_waist":     {"en": "Waist circumference", "hi": "कमर परिधि", "mr": "कंबर परिधि"},
    "ft_heart_disease":{"en":"Heart disease", "hi": "हृदय रोग", "mr": "हृदयरोग"},
    "ft_kidney_stones":{"en":"Kidney stones", "hi": "गुर्दे की पथरी", "mr": "किडनी स्टोन"},
    "note_high_bmi":{"en": "High BMI raises kidney risk — losing even 5–10% of body weight helps.", "hi": "उच्च BMI किडनी जोखिम बढ़ाता है।", "mr": "जास्त BMI किडनी धोका वाढवतो."},
    "note_young":  {"en": "Under 40 with risk factors: get a creatinine test once.", "hi": "40 से कम उम्र में जोखिम कारक: एक बार क्रिएटिनिन टेस्ट करें।", "mr": "40 पेक्षा कमी वयात धोका घटक: एकदा क्रिएटिनिन तपासणी करा."},
    # Yearly reminder
    "rem_title":   {"en": "Yearly reminder", "hi": "वार्षिक अनुस्मारक", "mr": "वार्षिक स्मरण"},
    "rem_due":     {"en": "Your yearly recheck is due soon.", "hi": "आपकी वार्षिक पुनर्जांच जल्द होने वाली है।", "mr": "तुमची वार्षिक पुनर्तपासणी लवकरच येत आहे."},
    "rem_text":    {"en": "Next check due: {d}", "hi": "अगली जांच: {d}", "mr": "पुढील तपासणी: {d}"},
    "rem_ics":     {"en": "Add to calendar (.ics)", "hi": "कैलेंडर में जोड़ें (.ics)", "mr": "कॅलेंडरमध्ये जोडा (.ics)"},
    # Privacy
    "priv_title":  {"en": "Privacy", "hi": "गोपनीयता", "mr": "गोपनीयता"},
    "del_btn":     {"en": "Delete all my home screening data", "hi": "मेरा सभी होम स्क्रीनिंग डेटा हटाएं", "mr": "माझा सर्व होम स्क्रीनिंग डेटा हटवा"},
    "del_done":    {"en": "Your home screening data has been deleted.", "hi": "आपका होम स्क्रीनिंग डेटा हटा दिया गया है।", "mr": "तुमचा होम स्क्रीनिंग डेटा हटवला गेला आहे."},
}


def t(lang: str, key: str, **kwargs) -> str:
    """Translate key to lang. Falls back to English. Formats kwargs into the string."""
    row = _T.get(key, {})
    text = row.get(lang) or row.get("en") or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text
