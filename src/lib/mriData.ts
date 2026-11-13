export interface MRICase {
  id: string;
  patientName: string;
  mrn: string; // Medical Record Number
  age: number;
  gender: "Male" | "Female";
  scanType: "Axial T1-Weighted Contrast Enhanced" | "Axial T2-FLAIR" | "Axial DWI";
  diagnosis: string;
  classification: "Glioma" | "Meningioma" | "Pituitary" | "Healthy";
  severity: "danger" | "warning" | "healthy";
  confidence: number; // e.g. 98.7
  tumorAreaMm2: number;
  tumorCentroid: { x: number; y: number; radius: number };
  brainRegion: string;
  massEffect: "Severe Midline Shift (6.2mm)" | "Mild Local Compression" | "None / Symmetrical";
  clinicalSummary: string;
  radiologistNotes: string;
  probabilities: {
    glioma: number;
    meningioma: number;
    pituitary: number;
    healthy: number;
  };
}

export const PRELOADED_MRI_CASES: MRICase[] = [
  {
    id: "case-gbm",
    patientName: "Patient #4082 (Arif M.)",
    mrn: "RAD-2026-004082",
    age: 54,
    gender: "Male",
    scanType: "Axial T1-Weighted Contrast Enhanced",
    diagnosis: "High-Grade Glioblastoma Multiforme (WHO Grade IV)",
    classification: "Glioma",
    severity: "danger",
    confidence: 98.7,
    tumorAreaMm2: 1280,
    tumorCentroid: { x: 390, y: 220, radius: 48 },
    brainRegion: "Right Frontotemporal Cortex",
    massEffect: "Severe Midline Shift (6.2mm)",
    clinicalSummary: "Heterogeneous infiltrative mass with irregular peripheral ring enhancement and central necrotic hypointensity.",
    radiologistNotes: "Severe vasogenic edema in surrounding white matter causing effacement of the ipsilateral lateral ventricle and 6.2mm subfalcine herniation.",
    probabilities: {
      glioma: 98.7,
      meningioma: 0.9,
      pituitary: 0.2,
      healthy: 0.2,
    },
  },
  {
    id: "case-meningioma",
    patientName: "Patient #2194 (Zainab K.)",
    mrn: "RAD-2026-002194",
    age: 48,
    gender: "Female",
    scanType: "Axial T1-Weighted Contrast Enhanced",
    diagnosis: "Atypical Parasagittal Meningioma (WHO Grade II)",
    classification: "Meningioma",
    severity: "danger",
    confidence: 94.2,
    tumorAreaMm2: 840,
    tumorCentroid: { x: 250, y: 170, radius: 38 },
    brainRegion: "Left Parasagittal Dural Convexity",
    massEffect: "Mild Local Compression",
    clinicalSummary: "Sharply circumscribed extra-axial lesion displaying vivid homogenous contrast enhancement with pathognomonic dural tail sign.",
    radiologistNotes: "Mass exerts compressive remodeling on adjacent parietal gyri without parenchymal infiltration. Superior sagittal sinus patent.",
    probabilities: {
      glioma: 3.8,
      meningioma: 94.2,
      pituitary: 1.2,
      healthy: 0.8,
    },
  },
  {
    id: "case-pituitary",
    patientName: "Patient #6819 (Bilal A.)",
    mrn: "RAD-2026-006819",
    age: 36,
    gender: "Male",
    scanType: "Axial T2-FLAIR",
    diagnosis: "Pituitary Macroadenoma (Sellar Neoplasm)",
    classification: "Pituitary",
    severity: "warning",
    confidence: 91.5,
    tumorAreaMm2: 420,
    tumorCentroid: { x: 320, y: 245, radius: 26 },
    brainRegion: "Sellar & Suprasellar Cistern",
    massEffect: "Mild Local Compression",
    clinicalSummary: "Expanded sella turcica containing an isointense mass extending superiorly into the suprasellar cistern.",
    radiologistNotes: "Lesion approaches the optic chiasm. Visual field Humphrey perimetry and endocrine panel correlation recommended.",
    probabilities: {
      glioma: 1.5,
      meningioma: 4.8,
      pituitary: 91.5,
      healthy: 2.2,
    },
  },
  {
    id: "case-healthy",
    patientName: "Patient #1105 (Nadia R.)",
    mrn: "RAD-2026-001105",
    age: 29,
    gender: "Female",
    scanType: "Axial T1-Weighted Contrast Enhanced",
    diagnosis: "Unremarkable Normal Brain MRI Scan",
    classification: "Healthy",
    severity: "healthy",
    confidence: 99.4,
    tumorAreaMm2: 0,
    tumorCentroid: { x: 0, y: 0, radius: 0 },
    brainRegion: "Intact Symmetrical Cerebral Hemispheres",
    massEffect: "None / Symmetrical",
    clinicalSummary: "Normal ventricular morphology with intact gray-white matter differentiation. Zero pathological contrast enhancement.",
    radiologistNotes: "Symmetrical lateral and third ventricles. Basal cisterns clear. No evidence of space-occupying lesion or acute ischemia.",
    probabilities: {
      glioma: 0.2,
      meningioma: 0.2,
      pituitary: 0.2,
      healthy: 99.4,
    },
  },
];
