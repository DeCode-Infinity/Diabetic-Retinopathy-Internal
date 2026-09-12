import { useState, useRef, useCallback } from "react";
import {
  Activity, Eye, FileText, Upload, Camera, AlertTriangle, CheckCircle2,
  Users, Hospital, RefreshCw, ChevronRight, Printer, Loader2, Info, X
} from "lucide-react";

/* ============================================================================
   AI-DR-Screen — connects directly to your trained EfficientNet-B4 model via
   inference_api.py. No mock data anywhere: grade, confidence, probabilities,
   quality metrics, the CLAHE image, and the Grad-CAM heatmap all come from
   the real backend response.

   Corrections vs. the reference mockup (flagging so nothing overclaims):
   - Tech badge says "EfficientNet-B4 · Grad-CAM" — not ResNet-50/Frangi
     Hessian, since that's not what's actually trained.
   - No "Frangi Vessels & Lesions" tab and no MA/exudate-count biomarker
     table — that needs the segmentation model, which hasn't been trained
     yet (still 0% per your roadmap). Shown as quality/confidence metrics
     instead, with an explicit note that lesion-level biomarkers are a
     Phase 2 item.
   - No Gemini "AI synthesis" — clinical impression / PHC guidance / patient
     summary are template text keyed by grade, not an LLM call, since no
     such integration is built. Labeled as clinical decision support, not
     multimodal AI synthesis.
   - No "Digitally Verifiable Seal" / QR claim — just a plain report ID.
   - No fake sample presets — only real uploaded images produce results.
============================================================================ */

const API_URL = "http://localhost:8000"; // point at your ngrok URL if backend isn't local

const GRADE_META = {
  0: {
    label: "No Diabetic Retinopathy", short: "Healthy", color: "emerald",
    triage: "Annual review (12 months)",
    impression: "No microaneurysms, haemorrhages, or exudates detected. Vascular pattern appears intact.",
    phc: "Reassure patient. Schedule next annual retinal photography screening. Reinforce glycemic control.",
    patient: "Your retinal scan looks healthy with no signs of diabetic eye damage right now. Keep managing your blood sugar and come back for your yearly check.",
  },
  1: {
    label: "Mild NPDR", short: "Mild", color: "sky",
    triage: "Re-screen in 6 months",
    impression: "Early-stage changes consistent with mild non-proliferative diabetic retinopathy.",
    phc: "No urgent referral needed. Schedule a follow-up screening in 6 months and reinforce glycemic control.",
    patient: "There are very early signs of diabetic eye changes. This is common and manageable — please come back in 6 months for a follow-up scan.",
  },
  2: {
    label: "Moderate NPDR", short: "Moderate", color: "amber",
    triage: "Refer to ophthalmologist (4–6 weeks)",
    impression: "Moderate non-proliferative diabetic retinopathy with multiple lesions present.",
    phc: "Refer to an ophthalmologist within 4–6 weeks. Continue monitoring glycemic status in the meantime.",
    patient: "Your scan shows moderate changes in your eyes related to diabetes. Please see an eye specialist within the next 4–6 weeks.",
  },
  3: {
    label: "Severe NPDR", short: "Severe", color: "orange",
    triage: "Urgent referral (1–2 weeks)",
    impression: "Severe non-proliferative diabetic retinopathy — high risk of progression.",
    phc: "Urgent referral required within 1–2 weeks. Flag this patient for priority follow-up.",
    patient: "Your scan shows significant changes that need urgent attention. Please see an eye specialist within the next 1–2 weeks.",
  },
  4: {
    label: "Proliferative DR", short: "PDR", color: "rose",
    triage: "IMMEDIATE referral",
    impression: "Proliferative diabetic retinopathy — signs of abnormal new vessel growth. Immediate risk to vision.",
    phc: "IMMEDIATE referral to an ophthalmologist. Do not delay — risk of vision loss.",
    patient: "Your scan shows serious changes that need immediate medical attention. Please see an eye specialist as soon as possible.",
  },
};

const COLOR_MAP = {
  emerald: { bg: "bg-emerald-500/10", text: "text-emerald-300", border: "border-emerald-500/30", dot: "bg-emerald-400", solid: "bg-emerald-500" },
  sky:     { bg: "bg-sky-500/10",     text: "text-sky-300",     border: "border-sky-500/30",     dot: "bg-sky-400",     solid: "bg-sky-500" },
  amber:   { bg: "bg-amber-500/10",   text: "text-amber-300",   border: "border-amber-500/30",   dot: "bg-amber-400",   solid: "bg-amber-500" },
  orange:  { bg: "bg-orange-500/10",  text: "text-orange-300",  border: "border-orange-500/30",  dot: "bg-orange-400",  solid: "bg-orange-500" },
  rose:    { bg: "bg-rose-500/10",    text: "text-rose-300",    border: "border-rose-500/30",    dot: "bg-rose-400",    solid: "bg-rose-500" },
};

// Real screening filter — rejects non-fundus uploads before they ever reach
// the model (same logic validated earlier against real/fake test images).
function checkIsRetina(imgEl) {
  const W = 200, H = 200;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.drawImage(imgEl, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);
  let rSum=0,gSum=0,bSum=0,aSum=0,cornerDark=0,n=0;
  const corners=[[5,5],[W-5,5],[5,H-5],[W-5,H-5]];
  for (let y=0;y<H;y+=4) for (let x=0;x<W;x+=4){
    const i=(y*W+x)*4; rSum+=data[i]; gSum+=data[i+1]; bSum+=data[i+2]; aSum+=data[i+3]; n++;
  }
  if (aSum/n < 250) return false; // transparent PNGs aren't photos
  corners.forEach(([x,y])=>{ const i=(y*W+x)*4; if((data[i]+data[i+1]+data[i+2])/3<40) cornerDark++; });
  const rAvg=rSum/n,gAvg=gSum/n,bAvg=bSum/n;
  return (rAvg>gAvg*1.15 && rAvg>bAvg*1.3) && cornerDark>=2;
}

const emptyPatient = {
  patient_id: "", patient_name: "", age: "", gender: "Female",
  diabetes_duration_years: "", hba1c: "", blood_pressure: "",
  eye_laterality: "OD (Right Eye)", phc_center_name: "", clinician_name: "",
};

export default function AIDRScreen() {
  const [patient, setPatient] = useState(emptyPatient);
  const [imgSrc, setImgSrc] = useState(null);
  const [stage, setStage] = useState("idle"); // idle | validating | processing | results | rejected | unsupported | apiError
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("original");
  const [view, setView] = useState("screening"); // screening | report | queue
  const [queue, setQueue] = useState([]);
  const [recapture, setRecapture] = useState(null); // {issues, quality} or null

  const fileInputRef = useRef(null);

  const setField = (field, value) => setPatient(p => ({ ...p, [field]: value }));

  const runScreening = useCallback((file, dataUrl) => {
    setStage("processing"); setError("");
    const formData = new FormData();
    formData.append("file", file);

    fetch(`${API_URL}/predict`, { method: "POST", body: formData })
      .then(async res => {
        if (res.status === 422) {
          const body = await res.json();
          setRecapture({ issues: body.detail?.issues || ["Image quality too low"], quality: body.detail?.quality });
          setStage("idle");
          return null;
        }
        if (!res.ok) throw new Error(`Server responded ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (!data) return;
        setResults(data);
        setImgSrc(dataUrl);
        setStage("results");
        setQueue(q => [{ id: Date.now(), patient: { ...patient }, results: data, imgSrc: dataUrl, ts: new Date() }, ...q]);
      })
      .catch(err => { setError(err.message || "Could not reach the model server."); setStage("apiError"); });
  }, [patient]);

  const handleFile = useCallback((file) => {
    if (!file?.type.startsWith("image/")) return;
    const rdr = new FileReader();
    rdr.onload = e => {
      const dataUrl = e.target.result;
      setStage("validating");
      const img = new Image();
      let settled = false;
      const timeout = setTimeout(() => { if (!settled) { settled = true; setImgSrc(dataUrl); setStage("unsupported"); } }, 4000);
      img.onerror = () => { if (settled) return; settled = true; clearTimeout(timeout); setImgSrc(dataUrl); setStage("unsupported"); };
      img.onload = () => {
        if (settled) return; settled = true; clearTimeout(timeout);
        if (!checkIsRetina(img)) { setImgSrc(dataUrl); setStage("rejected"); return; }
        runScreening(file, dataUrl);
      };
      img.src = dataUrl;
    };
    rdr.readAsDataURL(file);
  }, [runScreening]);

  const selectFromQueue = (entry) => {
    setPatient(entry.patient); setResults(entry.results); setImgSrc(entry.imgSrc);
    setStage("results"); setView("screening"); setTab("original");
  };

  const meta = results ? GRADE_META[results.grade] : null;
  const c = meta ? COLOR_MAP[meta.color] : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-4 py-3">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-emerald-400 flex items-center justify-center text-slate-950">
              <Eye className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-extrabold text-base text-white">AI-DR-Screen <span className="text-teal-400 font-mono text-xs">SIH 26038</span></h1>
              <p className="text-[11px] text-slate-400">Diabetic Retinopathy Screening for Rural Primary Health Centers</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
              {[["screening","Screening",Activity],["report","Report",FileText],["queue",`Session Queue (${queue.length})`,Users]].map(([id,label,Icon])=>(
                <button key={id} onClick={()=>setView(id)}
                  className={`px-3 py-1.5 rounded-md font-semibold flex items-center gap-1.5 transition-all ${view===id?"bg-teal-500 text-slate-950":"text-slate-400 hover:text-slate-200"}`}>
                  <Icon className="w-3.5 h-3.5" /><span>{label}</span>
                </button>
              ))}
            </div>
            <div className="hidden lg:flex items-center gap-2 bg-slate-900 px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-slate-300 font-mono">EfficientNet-B4 · Grad-CAM</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4 sm:p-6">
        {/* ══ SCREENING VIEW ══ */}
        {view === "screening" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: patient intake + upload */}
            <div className="lg:col-span-4 space-y-4">
              <div className="bg-slate-900/90 rounded-xl border border-slate-800 p-4 space-y-3">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-2.5">
                  <Hospital className="w-4 h-4 text-teal-400" />
                  <h3 className="text-xs font-bold text-slate-200">Patient Details</h3>
                </div>
                <div className="grid grid-cols-5 gap-2 text-xs">
                  <input value={patient.patient_id} onChange={e=>setField("patient_id",e.target.value)} placeholder="ID"
                    className="col-span-2 bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200 focus:border-teal-500 focus:outline-none" />
                  <input value={patient.patient_name} onChange={e=>setField("patient_name",e.target.value)} placeholder="Full name"
                    className="col-span-3 bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200 focus:border-teal-500 focus:outline-none" />
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex gap-1">
                    <input type="number" value={patient.age} onChange={e=>setField("age",e.target.value)} placeholder="Age"
                      className="w-1/2 bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200 focus:border-teal-500 focus:outline-none" />
                    <select value={patient.gender} onChange={e=>setField("gender",e.target.value)}
                      className="w-1/2 bg-slate-950 border border-slate-800 rounded px-1 py-1.5 text-slate-200 focus:border-teal-500 focus:outline-none">
                      <option>Female</option><option>Male</option><option>Other</option>
                    </select>
                  </div>
                  <input type="number" step="0.5" value={patient.diabetes_duration_years} onChange={e=>setField("diabetes_duration_years",e.target.value)} placeholder="DM years"
                    className="bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200 focus:border-teal-500 focus:outline-none" />
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <input type="number" step="0.1" value={patient.hba1c} onChange={e=>setField("hba1c",e.target.value)} placeholder="HbA1c %"
                    className="bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200 focus:border-teal-500 focus:outline-none font-mono" />
                  <input value={patient.blood_pressure} onChange={e=>setField("blood_pressure",e.target.value)} placeholder="BP e.g. 130/80"
                    className="bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-slate-200 focus:border-teal-500 focus:outline-none font-mono" />
                </div>
                <select value={patient.eye_laterality} onChange={e=>setField("eye_laterality",e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 focus:border-teal-500 focus:outline-none">
                  <option>OD (Right Eye)</option><option>OS (Left Eye)</option><option>Both Eyes</option>
                </select>
              </div>

              <div className="bg-slate-900/90 rounded-xl border border-slate-800 p-4 space-y-3">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-2.5">
                  <Camera className="w-4 h-4 text-teal-400" />
                  <h3 className="text-xs font-bold text-slate-200">Fundus Image</h3>
                </div>
                <div onClick={()=>fileInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-700 hover:border-teal-500 rounded-xl p-4 text-center cursor-pointer transition-all bg-slate-950/60 hover:bg-slate-950 flex flex-col items-center justify-center min-h-[140px]">
                  <input ref={fileInputRef} type="file" accept="image/*" className="hidden"
                    onChange={e=>{ const f=e.target.files?.[0]; if(f) handleFile(f); }} />
                  <div className="w-10 h-10 rounded-full bg-teal-500/10 text-teal-400 flex items-center justify-center mb-2">
                    <Upload className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-bold text-slate-200">Upload Retinal Fundus Scan</span>
                  <span className="text-[11px] text-slate-400 mt-1">JPG or PNG — click to browse</span>
                </div>
                {stage==="validating" && (
                  <div className="flex items-center gap-2 text-xs text-slate-400"><Loader2 className="w-3.5 h-3.5 animate-spin" />Checking image…</div>
                )}
                {stage==="processing" && (
                  <div className="flex items-center gap-2 text-xs text-teal-300"><Loader2 className="w-3.5 h-3.5 animate-spin" />Running AI screening…</div>
                )}
                {stage==="rejected" && (
                  <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/30 text-xs text-rose-300 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>Not a retinal fundus image. Please upload a genuine fundus photo.</span>
                  </div>
                )}
                {stage==="unsupported" && (
                  <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/30 text-xs text-amber-300 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>This format (e.g. .tif) can't be read by browsers. Convert to .jpg/.png first.</span>
                  </div>
                )}
                {stage==="apiError" && (
                  <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/30 text-xs text-rose-300">
                    <div className="flex items-start gap-2 mb-2"><AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" /><span>{error || "Couldn't reach the model server."}</span></div>
                    <button onClick={()=>fileInputRef.current?.click()} className="text-[11px] font-bold underline">Try again</button>
                  </div>
                )}
              </div>

              {results && stage==="results" && (
                <div className={`p-3 rounded-xl border flex items-center gap-2 text-xs ${results.quality>=70?"bg-emerald-950/30 border-emerald-500/30 text-emerald-300":"bg-amber-950/30 border-amber-500/30 text-amber-300"}`}>
                  {results.quality>=70 ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <Info className="w-4 h-4 shrink-0" />}
                  <span>Image quality: {results.quality}/100</span>
                </div>
              )}
            </div>

            {/* Right: image viewer + results */}
            <div className="lg:col-span-8 space-y-6">
              {stage!=="results" && !results && (
                <div className="bg-slate-900/60 rounded-xl border border-dashed border-slate-800 p-16 text-center text-slate-500">
                  <Eye className="w-8 h-8 mx-auto mb-3 text-slate-700" />
                  <p className="text-sm">Upload a fundus image to run a screening.</p>
                </div>
              )}

              {results && stage==="results" && (
                <>
                  <div className="bg-slate-900/90 rounded-xl border border-slate-800 p-4 space-y-3">
                    <div className="flex gap-1.5 border-b border-slate-800 pb-3">
                      {[["original","Original"],["enhanced","CLAHE Enhanced"],["heatmap","Grad-CAM"]].map(([id,label])=>(
                        <button key={id} onClick={()=>setTab(id)}
                          className={`px-3 py-1.5 rounded-md text-xs font-semibold ${tab===id?"bg-teal-500 text-slate-950":"bg-slate-950 text-slate-400 border border-slate-800"}`}>{label}</button>
                      ))}
                    </div>
                    <div className="rounded-xl overflow-hidden border border-slate-800 bg-black aspect-square relative">
                      <img src={tab==="enhanced"?results.enhanced:tab==="heatmap"?results.heatmap:imgSrc} alt="Fundus" className="w-full h-full object-cover" />
                    </div>
                  </div>

                  <div className={`rounded-xl border p-4 ${c.bg} ${c.border}`}>
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`w-2.5 h-2.5 rounded-full ${c.dot}`} />
                          <span className={`text-lg font-bold ${c.text}`}>Grade {results.grade}: {meta.label}</span>
                        </div>
                        <p className="text-xs text-slate-400">Confidence: {(results.confidence*100).toFixed(1)}%</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded text-[11px] font-bold ${c.bg} ${c.text} border ${c.border}`}>{meta.triage}</span>
                    </div>
                    <p className="text-xs text-slate-300 mt-3 leading-relaxed">{meta.impression}</p>
                  </div>

                  <div className="bg-slate-900/90 rounded-xl border border-slate-800 p-4 space-y-3">
                    <h3 className="text-xs font-bold text-slate-200">Grade Probability Distribution</h3>
                    {results.probs.map((p,i)=>(
                      <div key={i} className="flex items-center gap-3 text-xs">
                        <span className="w-24 text-slate-400 shrink-0">Grade {i} — {GRADE_META[i].short}</span>
                        <div className="flex-1 h-2 bg-slate-950 rounded-full overflow-hidden">
                          <div className={`h-full ${COLOR_MAP[GRADE_META[i].color].solid}`} style={{width:`${p*100}%`}} />
                        </div>
                        <span className="w-10 text-right font-mono text-slate-400">{(p*100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>

                  <div className="bg-slate-900/90 rounded-xl border border-slate-800 p-4 grid grid-cols-3 gap-3 text-xs">
                    <div><div className="text-slate-500 mb-1">Quality</div><div className="text-slate-200 font-mono text-sm">{results.quality}/100</div></div>
                    <div><div className="text-slate-500 mb-1">Sharpness</div><div className="text-slate-200 font-mono text-sm">{results.sharpness}</div></div>
                    <div><div className="text-slate-500 mb-1">Entropy</div><div className="text-slate-200 font-mono text-sm">{results.entropy}</div></div>
                  </div>
                  <p className="text-[11px] text-slate-500 -mt-3 px-1">Lesion-level biomarkers (microaneurysm/exudate counts, vessel density) require the segmentation model — planned for Phase 2.</p>

                  <button onClick={()=>setView("report")} className="w-full py-2.5 rounded-lg text-xs font-bold bg-teal-500 text-slate-950 hover:bg-teal-400 flex items-center justify-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" /><span>View Full Report</span><ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        {/* ══ REPORT VIEW ══ */}
        {view === "report" && results && meta && (
          <div className="max-w-3xl mx-auto bg-white text-slate-900 rounded-xl p-8 print:p-0 print:shadow-none shadow-2xl">
            <div className="flex items-center justify-between border-b-2 border-slate-900 pb-3 mb-4">
              <div>
                <h2 className="text-lg font-extrabold">AI-DR-SCREEN SCREENING REPORT</h2>
                <p className="text-xs text-slate-500">Rural PHC Diabetic Retinopathy Screening · SIH 26038</p>
              </div>
              <div className="text-right text-[11px] text-slate-500">
                <div>Report ID: {`DR-${(patient.patient_id||"XXXX")}-${Date.now().toString().slice(-6)}`}</div>
                <div>{new Date().toLocaleString()}</div>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-3 text-xs mb-4 pb-4 border-b border-slate-200">
              <div><div className="text-slate-500">Patient</div><div className="font-semibold">{patient.patient_name||"—"}</div></div>
              <div><div className="text-slate-500">Age / Gender</div><div className="font-semibold">{patient.age||"—"} / {patient.gender}</div></div>
              <div><div className="text-slate-500">Diabetes</div><div className="font-semibold">{patient.diabetes_duration_years||"—"} yrs · HbA1c {patient.hba1c||"—"}%</div></div>
              <div><div className="text-slate-500">Eye</div><div className="font-semibold">{patient.eye_laterality}</div></div>
            </div>

            <div className={`rounded-lg p-4 mb-4 border-2 ${meta.color==="emerald"?"border-emerald-600 bg-emerald-50":meta.color==="sky"?"border-sky-600 bg-sky-50":meta.color==="amber"?"border-amber-600 bg-amber-50":meta.color==="orange"?"border-orange-600 bg-orange-50":"border-rose-600 bg-rose-50"}`}>
              <div className="text-[11px] font-bold uppercase text-slate-500 mb-1">AI Grading Result</div>
              <div className="text-xl font-extrabold">Grade {results.grade}: {meta.label}</div>
              <div className="text-xs mt-1">Confidence: {(results.confidence*100).toFixed(1)}% · Triage: {meta.triage}</div>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4">
              {["original","enhanced","heatmap"].map(k=>(
                <div key={k} className="border border-slate-300 rounded-lg overflow-hidden">
                  <div className="text-[10px] font-bold uppercase text-center bg-slate-100 py-1">{k==="original"?"Original":k==="enhanced"?"Enhanced":"Grad-CAM"}</div>
                  <img src={k==="enhanced"?results.enhanced:k==="heatmap"?results.heatmap:imgSrc} className="w-full aspect-square object-cover" alt={k} />
                </div>
              ))}
            </div>

            <div className="mb-4">
              <div className="text-[11px] font-bold uppercase text-slate-500 mb-1">Clinical Impression</div>
              <p className="text-xs leading-relaxed">{meta.impression}</p>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                <div className="text-[10px] font-bold uppercase text-teal-700 mb-1">PHC Health Worker Guidance</div>
                <p className="text-[11px] leading-relaxed">{meta.phc}</p>
              </div>
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                <div className="text-[10px] font-bold uppercase text-amber-700 mb-1">Patient Summary</div>
                <p className="text-[11px] leading-relaxed">"{meta.patient}"</p>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-slate-200 text-[10px] text-slate-500">
              <span>AI-DR-Screen (SIH 26038) — screening decision support, not a clinical diagnosis</span>
              <button onClick={()=>window.print()} className="flex items-center gap-1 text-slate-700 font-semibold print:hidden">
                <Printer className="w-3 h-3" /> Print
              </button>
            </div>
          </div>
        )}
        {view === "report" && !results && (
          <div className="text-center text-slate-500 py-16 text-sm">Run a screening first to generate a report.</div>
        )}

        {/* ══ QUEUE VIEW ══ */}
        {view === "queue" && (
          <div className="max-w-4xl mx-auto space-y-3">
            <div className="bg-slate-900/90 rounded-xl border border-slate-800 p-4">
              <h2 className="text-sm font-bold flex items-center gap-2"><Users className="w-4 h-4 text-teal-400" />Session Queue</h2>
              <p className="text-xs text-slate-400 mt-1">Screenings run this session, sorted by urgency.</p>
            </div>
            {queue.length===0 && <div className="text-center text-slate-500 py-12 text-sm">No screenings yet this session.</div>}
            {[...queue].sort((a,b)=>b.results.grade-a.results.grade).map(entry=>{
              const m = GRADE_META[entry.results.grade]; const cc = COLOR_MAP[m.color];
              return (
                <button key={entry.id} onClick={()=>selectFromQueue(entry)}
                  className="w-full flex items-center justify-between bg-slate-900/90 rounded-xl border border-slate-800 hover:border-teal-500 p-3 text-left transition-all">
                  <div className="flex items-center gap-3">
                    <span className={`w-2.5 h-2.5 rounded-full ${cc.dot}`} />
                    <div>
                      <div className="text-sm font-semibold">{entry.patient.patient_name || "Unnamed patient"}</div>
                      <div className="text-[11px] text-slate-500">{entry.ts.toLocaleTimeString()}</div>
                    </div>
                  </div>
                  <span className={`px-2 py-1 rounded text-[11px] font-bold ${cc.bg} ${cc.text} border ${cc.border}`}>Grade {entry.results.grade} — {m.short}</span>
                </button>
              );
            })}
          </div>
        )}
      </main>

      {/* Recapture modal */}
      {recapture && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 border border-rose-500/30 rounded-xl p-5 max-w-sm w-full">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2 text-rose-300 font-bold text-sm"><AlertTriangle className="w-4 h-4" />Recapture Needed</div>
              <button onClick={()=>setRecapture(null)}><X className="w-4 h-4 text-slate-500" /></button>
            </div>
            <p className="text-xs text-slate-400 mb-2">Quality score: {recapture.quality}/100</p>
            <ul className="text-xs text-slate-300 list-disc pl-4 space-y-1 mb-4">
              {recapture.issues.map((iss,i)=><li key={i}>{iss}</li>)}
            </ul>
            <button onClick={()=>{ setRecapture(null); fileInputRef.current?.click(); }}
              className="w-full py-2 rounded-lg bg-teal-500 text-slate-950 text-xs font-bold flex items-center justify-center gap-1.5">
              <RefreshCw className="w-3.5 h-3.5" />Recapture Image
            </button>
          </div>
        </div>
      )}

      <footer className="border-t border-slate-800 px-4 py-3 text-center text-[11px] text-slate-500">
        AI-DR-Screen (SIH 26038) · Powered by EfficientNet-B4 (trained) · Real-time Grad-CAM explainability
      </footer>
    </div>
  );
}
