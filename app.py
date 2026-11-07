import os, numpy as np, pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
D = os.path.join(os.path.dirname(__file__), "data"); os.makedirs(D, exist_ok=True)
rng = np.random.RandomState(7)

def make_data():
    p = f"{D}/access_logs.csv"
    if not os.path.exists(p):
        n = 400; h = rng.randint(0, 24, n); b = rng.randint(0, 2, n); f = (rng.rand(n) < .25).astype(int)
        o = ((h < 6) | (h > 20)).astype(int)
        y = ((b + f + o) >= 2).astype(int)
        pd.DataFrame(dict(entry_hour=h, unknown_badge=b, forced_entry=f, outside_shift=o, unauthorized=y)).to_csv(p, index=False)
    p = f"{D}/signal_profiles.csv"
    if not os.path.exists(p):
        prof = {"TX-01": (88.5, 40, 35), "TX-02": (95.2, 62, 22), "TX-03": (99.8, 72, 14), "TX-04": (104.2, 81, 7), "TX-05": (107.1, 55, 30)}
        rows = [(rng.normal(f, .6), rng.normal(s, 4), rng.normal(nz, 3), k) for k, (f, s, nz) in prof.items() for _ in range(60)]
        pd.DataFrame(rows, columns=["frequency", "strength", "noise", "transmitter"]).to_csv(p, index=False)
    p = f"{D}/message_notes.csv"
    if not os.path.exists(p):
        sec = ["meet at the dock at midnight", "bring the tape tell no one", "burn this note after reading", "the recording must stay hidden",
               "do not trust the director", "delete the logs before dawn", "the package is hidden behind the cabinet", "keep quiet about the transmitter", "do not play the original recording", "check the backup at midnight", "the truth is hidden in the transmitter", "the backup recording is the only proof",
               "nobody can know about the broadcast", "destroy the tape tonight", "secret meeting after the shift", "hide the key and say nothing"]
        rou = ["weekly maintenance schedule attached", "please replace the broken microphone", "staff meeting moved to monday morning", "the coffee machine needs cleaning",
               "reminder to update the broadcast schedule", "new headphones ordered for the studio", "check the antenna cables on friday", "lunch menu for the canteen this week",
               "timesheet due by end of month", "fire drill planned for thursday", "invoice for the cleaning supplies", "please sign the visitor log", "please play the weekly weather report at nine", "check the backup generator on monday", "the studio schedule is on the notice board"]
        pd.DataFrame([(t + " " + w, "SECRET") for t in sec for w in ["", "tonight", "please"]] +
                     [(t + " " + w, "ROUTINE") for t in rou for w in ["", "thanks", "please"]], columns=["text", "label"]).to_csv(p, index=False)
make_data()
a = pd.read_csv(f"{D}/access_logs.csv"); AF = ["entry_hour", "unknown_badge", "forced_entry", "outside_shift"]
dt = DecisionTreeClassifier(max_depth=4, random_state=1).fit(a[AF], a.unauthorized)
s = pd.read_csv(f"{D}/signal_profiles.csv"); SF = ["frequency", "strength", "noise"]
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(s[SF], s.transmitter)
m = pd.read_csv(f"{D}/message_notes.csv")
nb = make_pipeline(CountVectorizer(), MultinomialNB()).fit(m.text, m.label)

app = Flask(__name__)
CORS(app)
@app.route("/")
def index(): return render_template("index.html")
@app.route("/api/health")
def health(): return jsonify(ok=True)

@app.route("/api/access", methods=["POST"])
def access():
    try:
        j = request.get_json(force=True); x = pd.DataFrame([[float(j[k]) for k in AF]], columns=AF)
        pr = int(dt.predict(x)[0]); imp = sorted(zip(AF, dt.feature_importances_), key=lambda t: -t[1])
        return jsonify(prediction="UNAUTHORIZED" if pr else "AUTHORIZED", factors=[dict(name=k, weight=round(float(v), 3)) for k, v in imp],
                       confidence=round(float(dt.predict_proba(x)[0].max()), 3))
    except Exception as e: return jsonify(error=str(e)), 400

@app.route("/api/signal", methods=["POST"])
def signal():
    try:
        j = request.get_json(force=True); x = pd.DataFrame([[float(j[k]) for k in SF]], columns=SF)
        pr = knn.predict(x)[0]; sc = knn.named_steps["standardscaler"]; k = knn.named_steps["kneighborsclassifier"]
        idx = k.kneighbors(sc.transform(x), return_distance=False)[0]
        return jsonify(transmitter=pr, neighbours=[s.transmitter.iloc[i] for i in idx], probabilities={c: round(float(p), 2) for c, p in zip(knn.classes_, knn.predict_proba(x)[0])})
    except Exception as e: return jsonify(error=str(e)), 400

@app.route("/api/message", methods=["POST"])
def message():
    try:
        t = str(request.get_json(force=True)["text"]); pr = nb.predict([t])[0]
        vec, clf = nb.named_steps["countvectorizer"], nb.named_steps["multinomialnb"]
        sec = list(clf.classes_).index("SECRET"); w = vec.get_feature_names_out()
        words = [(w[i], float(clf.feature_log_prob_[sec][i] - clf.feature_log_prob_[1 - sec][i])) for i in vec.transform([t]).nonzero()[1]]
        words.sort(key=lambda z: -z[1])
        return jsonify(label=pr, probabilities={c: round(float(p), 3) for c, p in zip(nb.classes_, nb.predict_proba([t])[0])}, suspicious_words=[a for a, b in words[:4] if b > 0])
    except Exception as e: return jsonify(error=str(e)), 400

if __name__ == "__main__": app.run(debug=True)
