"use client";

import React from "react";
import { MRICase } from "@/lib/mriData";
import { Printer, X, ShieldCheck, CheckCircle2, AlertOctagon } from "lucide-react";

interface Props {
  mriCase: MRICase;
  onClose: () => void;
}

export const RadiologyReportModal: React.FC<Props> = ({ mriCase, onClose }) => {
  const handlePrint = () => {
    window.print();
  };

  const isMalignant = mriCase.severity === "danger";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md">
      <div className="relative w-full max-w-3xl max-h-[92vh] overflow-y-auto bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-6 sm:p-8">
        {/* Top Modal Controls */}
        <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <span className="flex h-3 w-3 rounded-full bg-cyan-400 animate-ping" />
            <span className="text-xs font-mono font-semibold tracking-wider text-cyan-300 uppercase">
              Clinical PACS Diagnostic Report Generator
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={handlePrint}
              className="flex items-center px-3.5 py-1.5 text-xs font-semibold text-slate-100 bg-cyan-600 hover:bg-cyan-500 rounded-lg shadow-md transition-colors"
            >
              <Printer className="w-3.5 h-3.5 mr-1.5" />
              Print / Save Clinical PDF
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable Hospital Radiology Document */}
        <div
          id="printable-report"
          className="p-8 bg-white text-slate-900 rounded-xl shadow-lg font-sans text-xs leading-relaxed border border-slate-300"
        >
          {/* Hospital Header */}
          <div className="text-center pb-4 mb-6 border-b-2 border-slate-900">
            <h1 className="text-xl sm:text-2xl font-black tracking-wide uppercase text-slate-950 font-serif">
              NeuroScan Advanced Radiology Institute
            </h1>
            <p className="text-xs text-slate-600">
              Department of Neuro-Oncology & Explainable Artificial Intelligence (XAI)
            </p>
            <p className="text-[11px] text-slate-500">
              DICOM PACS Network • Accredited Radiology AI System • Ref: NEURO-AI-2026-X
            </p>
          </div>

          {/* Patient Demographic Table */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 bg-slate-100 border border-slate-300 rounded-lg mb-6 text-xs">
            <div>
              <span className="text-[10px] text-slate-500 block uppercase">Patient Identifier</span>
              <span className="font-bold text-slate-900">{mriCase.patientName}</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 block uppercase">Medical Record #</span>
              <span className="font-mono font-bold text-slate-900">{mriCase.mrn}</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 block uppercase">Age / Gender</span>
              <span className="font-semibold text-slate-900">{mriCase.age} Years • {mriCase.gender}</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-500 block uppercase">Exam Sequence</span>
              <span className="font-semibold text-slate-900">{mriCase.scanType}</span>
            </div>
          </div>

          {/* Primary AI Classification Headline */}
          <div className="mb-6 p-4 rounded-xl border-l-4 border-l-red-600 bg-red-50 border border-red-200">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-bold uppercase tracking-wider text-red-900 flex items-center space-x-1.5">
                {isMalignant ? <AlertOctagon className="w-4 h-4 text-red-600" /> : <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
                <span>AI Deep Learning Diagnostic Classification:</span>
              </span>
              <span className="font-mono font-bold text-xs px-2 py-0.5 rounded bg-red-200 text-red-900">
                Confidence: {mriCase.confidence}%
              </span>
            </div>
            <h2 className="text-base sm:text-lg font-black text-slate-950 font-serif">
              {mriCase.diagnosis}
            </h2>
          </div>

          {/* Quantitative Tumor Biometrics */}
          <div className="mb-6">
            <h3 className="text-xs font-bold uppercase text-slate-800 tracking-wider mb-2 border-b pb-1">
              Quantitative Lesion Biometrics & Localization
            </h3>
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="p-2.5 bg-slate-50 border rounded-lg">
                <span className="text-[10px] text-slate-500 block">Lesion Site</span>
                <span className="font-bold text-slate-900">{mriCase.brainRegion}</span>
              </div>
              <div className="p-2.5 bg-slate-50 border rounded-lg">
                <span className="text-[10px] text-slate-500 block">Tumor Surface Area</span>
                <span className="font-bold text-slate-900">{mriCase.tumorAreaMm2} mm²</span>
              </div>
              <div className="p-2.5 bg-slate-50 border rounded-lg">
                <span className="text-[10px] text-slate-500 block">Mass Effect Index</span>
                <span className="font-bold text-slate-900">{mriCase.massEffect}</span>
              </div>
            </div>
          </div>

          {/* Radiological Findings */}
          <div className="mb-6 space-y-2">
            <h3 className="text-xs font-bold uppercase text-slate-800 tracking-wider border-b pb-1">
              Radiological Findings & Anatomical Description
            </h3>
            <p className="text-slate-800 text-xs sm:text-sm leading-relaxed text-justify">
              {mriCase.clinicalSummary}
            </p>
            <p className="text-slate-800 text-xs sm:text-sm leading-relaxed text-justify">
              {mriCase.radiologistNotes}
            </p>
          </div>

          {/* Explainable AI Grad-CAM Verification Box */}
          <div className="mb-8 p-3.5 bg-cyan-50 border border-cyan-200 rounded-lg">
            <div className="flex items-center space-x-1.5 text-cyan-900 font-bold text-xs mb-1">
              <ShieldCheck className="w-4 h-4 text-cyan-700" />
              <span>Explainable AI (XAI) Grad-CAM Verification</span>
            </div>
            <p className="text-[11px] text-slate-700 leading-normal">
              Neural activation was derived via DenseNet-121 gradient backpropagation across the final convolutional block (<span className="font-mono">conv5_block3_out</span>). The gradient attribution vector correlates with pathological microvascular proliferation and hyperintense peripheral enhancement.
            </p>
          </div>

          {/* Signatures */}
          <div className="pt-6 border-t border-slate-300 grid grid-cols-2 gap-8 text-xs font-sans">
            <div>
              <p className="font-bold text-slate-900">Dr. Assif A. Dar, MD</p>
              <p className="text-slate-500 text-[11px]">Senior Consultant Neuro-Radiologist</p>
              <p className="text-slate-400 text-[10px]">Board of Nuclear Medicine & Radiology</p>
            </div>
            <div className="text-right">
              <p className="font-bold text-slate-900">Dar Rehan Rasool, B.Tech</p>
              <p className="text-slate-500 text-[11px]">Lead AI Diagnostic System Architect</p>
              <p className="text-slate-400 text-[10px]">Medical Imaging Machine Learning Laboratory</p>
            </div>
          </div>
        </div>

        {/* Modal Bottom Action */}
        <div className="mt-5 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
          >
            Close Viewer
          </button>
        </div>
      </div>
    </div>
  );
};
