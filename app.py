import streamlit as st
import numpy as np
import cv2
from PIL import Image
import uuid
import json, os, hashlib

st.set_page_config(page_title="DeepTrust AI", layout="wide")

# ─── STYLE ───
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color:white;
}
h1,h2,h3 { text-align:center; }
</style>
""", unsafe_allow_html=True)

# ─── USER SYSTEM ───
USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    return json.load(open(USER_FILE))

def save_users(u):
    json.dump(u, open(USER_FILE,"w"))

def hash_pass(p):
    return hashlib.sha256(p.encode()).hexdigest()

users = load_users()

# ─── SESSION ───
if "logged" not in st.session_state:
    st.session_state.logged = False

if "history" not in st.session_state:
    st.session_state.history = []

# ─── AUTH ───
def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u] == hash_pass(p):
            st.session_state.logged = True
            st.session_state.user = u
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

def signup():
    st.subheader("📝 Sign Up")
    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if u in users:
            st.warning("User exists")
        else:
            users[u] = hash_pass(p)
            save_users(users)
            st.success("Account created")

# ─── AUTH FLOW ───
if not st.session_state.logged:
    st.title("🛡️ DeepTrust AI")
    tab1, tab2 = st.tabs(["Login","Signup"])
    with tab1: login()
    with tab2: signup()
    st.stop()

# ─── LOGOUT ───
with st.sidebar:
    st.write(f"👤 {st.session_state.user}")
    if st.button("Logout"):
        st.session_state.logged = False
        st.rerun()

st.title("🛡️ DeepTrust AI")
st.caption("Final Hackathon Build")

# ─── DETECTOR ───
class Detector:
    def analyze(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Features
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise = np.std(gray)
        edges = np.mean(cv2.Canny(gray,100,200))

        # ✅ Proper normalization (FIXED)
        texture_score = min(texture / 150, 1)
        noise_score = min(noise / 70, 1)
        edge_score = min(edges / 100, 1)

        score = (texture_score * 0.4 +
                 noise_score * 0.3 +
                 edge_score * 0.3)

        score = int(score * 100)

        # Face detection
        face = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face.detectMultiScale(gray, 1.3, 5)

        # ✅ Smart correction
        if len(faces) == 0:
            score = 75
        else:
            score += 5

        score = max(0, min(score, 100))

        # ✅ Better thresholds
        if score >= 70:
            verdict = "Likely Real ✅"
        elif score >= 50:
            verdict = "Uncertain ⚠️"
        else:
            verdict = "Likely Fake 🚨"

        return score, verdict

detector = Detector()

# ─── HEATMAP ───
def gradcam_like(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F,1,0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F,0,1)

    heat = np.sqrt(grad_x**2 + grad_y**2)
    heat = cv2.normalize(heat,None,0,255,cv2.NORM_MINMAX)
    heat = np.uint8(heat)

    heat = cv2.GaussianBlur(heat,(15,15),0)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heat,0.4,0)

    return heat, overlay

# ─── MODES ───
mode = st.sidebar.radio("Mode", ["Upload","Compare","Dashboard"])

# ─── UPLOAD ───
if mode == "Upload":

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)

        st.image(file, use_column_width=True)

        if st.button("Analyze 🚀"):
            score, verdict = detector.analyze(img)

            st.session_state.history.append(score)

            st.progress(score/100)
            st.subheader(f"{verdict} ({score})")

            heat, overlay = gradcam_like(img)

            col1, col2 = st.columns(2)
            col1.image(heat)
            col2.image(overlay)

            with st.expander("🧠 Explanation"):
                st.write("Analyzing texture, noise, and edge consistency.")

            st.code(f"Verification ID: {str(uuid.uuid4())[:8]}")

# ─── COMPARE ───
elif mode == "Compare":

    col1, col2 = st.columns(2)

    f1 = col1.file_uploader("Image 1", key="1")
    f2 = col2.file_uploader("Image 2", key="2")

    if f1 and f2:
        img1 = cv2.cvtColor(np.array(Image.open(f1)), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(Image.open(f2)), cv2.COLOR_RGB2BGR)

        col1.image(f1)
        col2.image(f2)

        if st.button("Compare 🚀"):
            s1,v1 = detector.analyze(img1)
            s2,v2 = detector.analyze(img2)

            col1.subheader(f"{v1} ({s1})")
            col2.subheader(f"{v2} ({s2})")

            if abs(s1-s2)<10:
                st.warning("Both images similar")
            elif s1>s2:
                st.success("Image 1 more authentic")
            else:
                st.success("Image 2 more authentic")

# ─── DASHBOARD ───
elif mode == "Dashboard":

    st.title("📊 Analytics Dashboard")

    data = st.session_state.history

    if len(data)==0:
        st.info("No data yet")
    else:
        st.line_chart(data)

        real = sum(1 for x in data if x>70)
        fake = sum(1 for x in data if x<40)

        st.bar_chart({"Real":real,"Fake":fake})

# ─── FOOTER ───
st.markdown("---")
st.caption("🚀 DeepTrust AI | Final Version")
