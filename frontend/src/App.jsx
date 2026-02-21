import { useState } from "react";

export default function App() {
  const [store, setStore] = useState(1);
  const [date, setDate] = useState("2015-09-17");
  const [promo, setPromo] = useState(1);
  const [stateHoliday, setStateHoliday] = useState("0");
  const [schoolHoliday, setSchoolHoliday] = useState(0);

  const [loading, setLoading] = useState(false);
  const [pred, setPred] = useState(null);
  const [error, setError] = useState("");

  async function onPredict() {
    setLoading(true);
    setError("");
    setPred(null);

    const payload = {
      rows: [
        {
          Store: Number(store),
          Date: date, // YYYY-MM-DD
          Promo: Number(promo),
          StateHoliday: stateHoliday,
          SchoolHoliday: Number(schoolHoliday),
        },
      ],
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt}`);
      }

      const data = await res.json();
      setPred(data.predictions?.[0] ?? null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: 16, fontFamily: "Arial" }}>
      <h1>Rossmann Sales Predictor</h1>
      <p>Frontend (React) → Backend (FastAPI) → Model</p>

      <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
        <Field label="Store">
          <input type="number" min="1" value={store} onChange={(e) => setStore(e.target.value)} />
        </Field>

        <Field label="Date">
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </Field>

        <Field label="Promo">
          <select value={promo} onChange={(e) => setPromo(e.target.value)}>
            <option value={0}>0</option>
            <option value={1}>1</option>
          </select>
        </Field>

        <Field label="StateHoliday">
          <select value={stateHoliday} onChange={(e) => setStateHoliday(e.target.value)}>
            <option value="0">0</option>
            <option value="a">a</option>
            <option value="b">b</option>
            <option value="c">c</option>
          </select>
        </Field>

        <Field label="SchoolHoliday">
          <select value={schoolHoliday} onChange={(e) => setSchoolHoliday(e.target.value)}>
            <option value={0}>0</option>
            <option value={1}>1</option>
          </select>
        </Field>

        <button onClick={onPredict} disabled={loading} style={{ padding: "10px 14px", cursor: "pointer" }}>
          {loading ? "Predicting..." : "Predict"}
        </button>

        <div style={{ marginTop: 14 }}>
          {pred !== null && (
            <h2 style={{ margin: 0 }}>Predicted Sales: {Number(pred).toFixed(2)}</h2>
          )}
          {error && <p style={{ color: "crimson" }}>{error}</p>}
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 12, marginBottom: 10, alignItems: "center" }}>
      <div>{label}</div>
      {children}
    </div>
  );
}