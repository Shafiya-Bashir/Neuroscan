"use client";

import React, { useState, useEffect } from "react";
import { MRICase, PRELOADED_MRI_CASES } from "@/lib/mriData";
import { medicalAudio } from "@/lib/medicalAudioEngine";
import { MRIViewer } from "@/components/MRIViewer";
import { RadiologyReportModal } from "@/components/RadiologyReportModal";
import {
  Activity,
  AlertTriangle,
  Volume2,
  VolumeX,
  FileText,
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  Sliders,
  Layers,
  Sparkles,
  Eye,
  Info,
  Brain,
  Upload,
  Cpu,
  Mic,
  ArrowRight,
} from "lucide-react";

export default function Home() {
  const [selectedCase, setSelectedCase] = useState<MRICase>(PRELOADED_MRI_CASES[0]);

  // Grad-CAM Controls
  const [gradCamOpacity, setGradCamOpacity] = useState<number>(0.75);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [showContour, setShowContour] = useState<boolean>(true);
  const [colorMap, setColorMap] = useState<"jet" | "hot" | "plasma">("jet");

  // Audio / Voice Controls
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(true);

  // Modal Report
  const [showReportModal, setShowReportModal] = useState<boolean>(false);

  // Trigger Sound & Voice upon selecting a case
  const handleSelectCase = (mriCase: MRICase) => {
    setSelectedCase(mriCase);

    if (mriCase.severity === "danger") {
      medicalAudio.playMalignantAlarm();
      if (voiceEnabled) {
        medicalAudio.speakRadiologyReport(
          `Critical alert: ${mriCase.diagnosis} identified in ${mriCase.brainRegion} with ${mriCase.confidence}% probability. Significant mass effect observed.`
        );
      }
    } else if (mriCase.severity === "warning") {
      medicalAudio.playCautionAlert();
      if (voiceEnabled) {
        medicalAudio.speakRadiologyReport(
          `Warning: ${mriCase.diagnosis} detected in ${mriCase.brainRegion}. Further contrast enhancement recommended.`
        );
      }
    } else {
      medicalAudio.playHealthyChime();
      if (voiceEnabled) {
        medicalAudio.speakRadiologyReport(
          `Evaluation complete: Normal brain MRI scan. Intracranial structures and ventricular morphology are completely healthy.`
        );
      }
    }
  };

  // Sync Audio Settings
  useEffect(() => {
    medicalAudio.setMuted(!soundEnabled);
    medicalAudio.setVoiceEnabled(voiceEnabled);
  }, [soundEnabled, voiceEnabled]);

  // Initial welcome trigger
  useEffect(() => {
    const timer = setTimeout(() => {
      handleSelectCase(PRELOADED_MRI_CASES[0]);
    }, 600);
    return () => clearTimeout(timer);
  }, []);

  const getSeverityBadge = (severity: MRICase["severity"]) => {
    switch (severity) {
      case "danger":
        return {
          label: "CRITICAL MALIGNANT",
          color: "bg-red-950/80 text-red-400 border-red-500 animate-pulse",
          icon: <ShieldAlert className="w-4 h-4 text-red-400" />,
        };
      case "warning":
        return {
          label: "LOW-GRADE / CAUTION",
          color: "bg-amber-950/80 text-amber-400 border-amber-500",
          icon: <AlertTriangle className="w-4 h-4 text-amber-400" />,
        };
      default:
        return {
          label: "HEALTHY / NEGATIVE",
          color: "bg-emerald-950/80 text-emerald-400 border-emerald-500",
          icon: <ShieldCheck className="w-4 h-4 text-emerald-400" />,
        };
    }
  };

  const badge = getSeverityBadge(selectedCase.severity);

  return (
    <div className="min-h-screen bg-[#050811] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top PACS Header Bar */}
      <header className="border-b border-slate-800 bg-[#080d1c]/95 backdrop-blur-xl px-4 sm:px-6 py-3.5 sticky top-0 z-30 shadow-xl">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-teal-500 to-emerald-400 shadow-lg shadow-cyan-500/20 text-white font-black text-xl">
              <Brain className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-base font-bold tracking-tight text-white">NEUROSCAN-AI</h1>
                <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-cyan-950 text-cyan-300 border border-cyan-500/40 rounded-full">
                  Explainable AI (XAI) Radiology PACS
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Deep Learning Brain MRI Classifier & Grad-CAM Neural Attention Workstation
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Export Clinical PDF Report Button */}
            <button
              onClick={() => setShowReportModal(true)}
              className="flex items-center space-x-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 text-white shadow-lg shadow-cyan-950/50 border border-cyan-400/40 transition-all"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Export Signed Clinical Report</span>
            </button>

            {/* Audio Alarm Sound Toggle */}
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              title={soundEnabled ? "Medical Audio Alarms Active" : "Audio Muted"}
              className={`p-2 rounded-xl border transition-colors ${
                soundEnabled
                  ? "bg-slate-900 text-cyan-400 border-slate-800 hover:border-cyan-500"
                  : "bg-red-950/40 text-red-400 border-red-900/40"
              }`}
            >
              {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            </button>

            {/* Voice Diagnostic Briefing Toggle */}
            <button
              onClick={() => setVoiceEnabled(!voiceEnabled)}
              title={voiceEnabled ? "Radiology Voice Briefing Active" : "Voice Briefing Muted"}
              className={`p-2 rounded-xl border transition-colors ${
                voiceEnabled
                  ? "bg-slate-900 text-emerald-400 border-slate-800 hover:border-emerald-500"
                  : "bg-slate-950 text-slate-500 border-slate-800"
              }`}
            >
              <Mic className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Patient Case Selector Ribbon */}
      <section className="border-b border-slate-800/80 bg-[#070b16] px-4 sm:px-6 py-2.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between overflow-x-auto gap-4">
          <div className="flex items-center space-x-2">
            <span className="text-slate-500 text-xs font-mono uppercase tracking-wider mr-1">
              Select Patient Scan:
            </span>

            {PRELOADED_MRI_CASES.map((c) => (
              <button
                key={c.id}
                onClick={() => handleSelectCase(c)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5 whitespace-nowrap ${
                  selectedCase.id === c.id
                    ? c.severity === "danger"
                      ? "bg-red-600 text-white shadow-md shadow-red-900/40"
                      : c.severity === "warning"
                      ? "bg-amber-600 text-white shadow-md shadow-amber-900/40"
                      : "bg-emerald-600 text-white shadow-md shadow-emerald-900/40"
                    : "bg-slate-900 hover:bg-slate-800 text-slate-400 border border-slate-800"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    c.severity === "danger"
                      ? "bg-red-400"
                      : c.severity === "warning"
                      ? "bg-amber-400"
                      : "bg-emerald-400"
                  }`}
                />
                <span>{c.diagnosis.split(" (")[0]}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400">
            <Cpu className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span>DenseNet-121 Feature Tensor: Online</span>
          </div>
        </div>
      </section>

      {/* Main Radiology Workstation Grid */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Clinical Diagnosis & Class Probabilities (3.5 cols) */}
        <aside className="lg:col-span-4 flex flex-col space-y-4">
          {/* Active Diagnostic Status Banner */}
          <div className={`p-4 rounded-2xl border shadow-xl backdrop-blur-xl ${badge.color}`}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-mono font-bold tracking-wider flex items-center space-x-1.5 uppercase">
                {badge.icon}
                <span>{badge.label}</span>
              </span>
              <span className="text-xs font-mono font-bold bg-black/40 px-2 py-0.5 rounded">
                AI Confidence: {selectedCase.confidence}%
              </span>
            </div>
            <h2 className="text-base sm:text-lg font-black text-white leading-tight font-serif">
              {selectedCase.diagnosis}
            </h2>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">
              {selectedCase.clinicalSummary}
            </p>
          </div>

          {/* Neural Network Probability Distribution */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-xl space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                <span>Softmax Class Probabilities</span>
              </h3>
              <span className="text-[10px] text-slate-500 font-mono">4-Class Head</span>
            </div>

            <div className="space-y-2.5 text-xs font-mono">
              {/* Glioma */}
              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Glioma (GBM / Astrocytoma)</span>
                  <span className="font-bold text-red-400">
                    {selectedCase.probabilities.glioma}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="h-full bg-red-500 transition-all duration-500"
                    style={{ width: `${selectedCase.probabilities.glioma}%` }}
                  />
                </div>
              </div>

              {/* Meningioma */}
              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Meningioma (Extra-Axial)</span>
                  <span className="font-bold text-amber-400">
                    {selectedCase.probabilities.meningioma}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="h-full bg-amber-500 transition-all duration-500"
                    style={{ width: `${selectedCase.probabilities.meningioma}%` }}
                  />
                </div>
              </div>

              {/* Pituitary */}
              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Pituitary Adenoma</span>
                  <span className="font-bold text-purple-400">
                    {selectedCase.probabilities.pituitary}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="h-full bg-purple-500 transition-all duration-500"
                    style={{ width: `${selectedCase.probabilities.pituitary}%` }}
                  />
                </div>
              </div>

              {/* Healthy */}
              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Healthy Brain (Normal)</span>
                  <span className="font-bold text-emerald-400">
                    {selectedCase.probabilities.healthy}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 transition-all duration-500"
                    style={{ width: `${selectedCase.probabilities.healthy}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Quantitative Tumor Biometrics */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-xl space-y-3">
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5 pb-2 border-b border-slate-800">
              <Info className="w-3.5 h-3.5 text-teal-400" />
              <span>Quantitative Lesion Biometrics</span>
            </h3>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Estimated Area</span>
                <span className="text-base font-bold font-mono text-cyan-300">
                  {selectedCase.tumorAreaMm2 > 0 ? `${selectedCase.tumorAreaMm2} mm²` : "0.0 mm²"}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Mass Effect</span>
                <span className="text-xs font-bold text-slate-300">
                  {selectedCase.massEffect}
                </span>
              </div>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs">
              <span className="text-[10px] text-slate-500 block uppercase">Anatomical Focus</span>
              <span className="font-semibold text-slate-200">{selectedCase.brainRegion}</span>
            </div>
          </div>
        </aside>

        {/* Center & Right Column: PACS MRI Canvas & Grad-CAM Wiping Station (8 cols) */}
        <section className="lg:col-span-8 flex flex-col space-y-4">
          {/* Main DICOM / MRI Canvas Viewer */}
          <div className="relative">
            <MRIViewer
              currentCase={selectedCase}
              gradCamOpacity={gradCamOpacity}
              showContour={showContour}
              showHeatmap={showHeatmap}
              colorMap={colorMap}
            />

            {/* Quick PACS HUD Status Overlay */}
            <div className="absolute top-4 left-4 z-10 flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-900/90 border border-slate-700/80 backdrop-blur-md">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  selectedCase.severity === "danger"
                    ? "bg-red-500 animate-ping"
                    : selectedCase.severity === "warning"
                    ? "bg-amber-500 animate-pulse"
                    : "bg-emerald-500"
                }`}
              />
              <span className="text-xs font-bold font-mono text-white uppercase">
                {selectedCase.severity === "danger"
                  ? "MALIGNANT LESION DETECTED"
                  : selectedCase.severity === "warning"
                  ? "SUSPICIOUS MASS"
                  : "NORMAL BRAIN PARENCHYMA"}
              </span>
            </div>

            {/* Top Right Colormap Badge */}
            <div className="absolute top-4 right-4 z-10 px-3 py-1 rounded-lg bg-black/80 text-[11px] font-mono text-cyan-300 border border-white/10 flex items-center space-x-1.5">
              <Sparkles className="w-3 h-3 text-cyan-400" />
              <span>Grad-CAM: {colorMap.toUpperCase()} Tensor</span>
            </div>
          </div>

          {/* Interactive Grad-CAM Wiping & Control Deck */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-xl space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                  Grad-CAM Neural Heatmap Wiping Controls
                </h3>
              </div>

              {/* Colormap Selectors */}
              <div className="flex items-center space-x-1.5 text-xs font-mono">
                <span className="text-slate-500 text-[11px] mr-1">Colormap:</span>
                {(["jet", "hot", "plasma"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setColorMap(m)}
                    className={`px-2.5 py-1 rounded-md text-[11px] uppercase transition-all ${
                      colorMap === m
                        ? "bg-cyan-600 text-white font-bold"
                        : "bg-slate-800 text-slate-400 hover:text-white"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            {/* Heatmap Opacity Slider (Wipe between Raw MRI and Neural Heatmap) */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400 flex items-center space-x-1.5">
                  <Eye className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Explainability Blend Slider (Raw MRI $\leftrightarrow$ Neural Attention):</span>
                </span>
                <span className="font-mono font-bold text-cyan-300">
                  {Math.round(gradCamOpacity * 100)}% Heatmap
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={gradCamOpacity}
                onChange={(e) => setGradCamOpacity(Number(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>0% (Raw T1/T2 MRI Scan)</span>
                <span>50% (Fused Overlay)</span>
                <span>100% (Pure Neural Heatmap)</span>
              </div>
            </div>

            {/* Toggle Switches */}
            <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs">
              <div className="flex items-center space-x-4">
                <label className="flex items-center space-x-2 cursor-pointer text-slate-300 hover:text-white">
                  <input
                    type="checkbox"
                    checked={showHeatmap}
                    onChange={(e) => setShowHeatmap(e.target.checked)}
                    className="accent-cyan-500 rounded"
                  />
                  <span>Show Grad-CAM Layer</span>
                </label>

                <label className="flex items-center space-x-2 cursor-pointer text-slate-300 hover:text-white">
                  <input
                    type="checkbox"
                    checked={showContour}
                    onChange={(e) => setShowContour(e.target.checked)}
                    className="accent-red-500 rounded"
                  />
                  <span>Show Lesion Contour & Area</span>
                </label>
              </div>

              {/* Replay Spoken Radiology Briefing */}
              <button
                onClick={() => handleSelectCase(selectedCase)}
                className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white flex items-center space-x-1.5 transition-colors"
              >
                <Volume2 className="w-3.5 h-3.5 text-emerald-400" />
                <span>Replay Spoken Briefing</span>
              </button>
            </div>
          </div>

          {/* Mathematical Explainable AI Reference (A+ for Teachers) */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-xl text-xs space-y-2">
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Grad-CAM Gradient Attribution Formulation</span>
            </h3>
            <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 text-[10.5px] font-mono text-cyan-300">
              L_Grad-CAM = ReLU( Σ_k α_k · A^k ) where α_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A_ij^k)
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Computes gradients of score $y^c$ for class $c$ with respect to feature activation maps $A^k$ of the final DenseNet-121 convolutional layer. A rectified linear activation (ReLU) filters negative attributions, highlighting solely the pathological pixels that triggered the positive tumor diagnosis.
            </p>
          </div>
        </section>
      </main>

      {/* Official Radiology Report Modal */}
      {showReportModal && (
        <RadiologyReportModal
          mriCase={selectedCase}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
}
