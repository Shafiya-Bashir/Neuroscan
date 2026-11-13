"use client";

import React, { useEffect, useRef, useState } from "react";
import { MRICase } from "@/lib/mriData";

interface Props {
  currentCase: MRICase;
  gradCamOpacity: number; // 0 to 1
  showContour: boolean;
  showHeatmap: boolean;
  colorMap: "jet" | "hot" | "plasma";
}

export const MRIViewer: React.FC<Props> = ({
  currentCase,
  gradCamOpacity,
  showContour,
  showHeatmap,
  colorMap,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // 1. Draw Authentic Axial Brain MRI anatomy
    drawAxialBrainMRI(ctx, width, height, currentCase);

    // 2. Draw Grad-CAM Neural Activation Heatmap
    if (showHeatmap && currentCase.classification !== "Healthy" && gradCamOpacity > 0) {
      drawGradCAMHeatmap(ctx, width, height, currentCase, gradCamOpacity, colorMap);
    }

    // 3. Draw Lesion Contour & Bounding Reticle
    if (showContour && currentCase.classification !== "Healthy") {
      drawLesionContour(ctx, currentCase);
    }

    // 4. Draw HUD Overlays & PACS Telemetry
    drawPacsTelemetry(ctx, width, height, currentCase, mousePos);
  }, [currentCase, gradCamOpacity, showContour, showHeatmap, colorMap, mousePos]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 640);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 540);
    setMousePos({ x, y });
  };

  const handleMouseLeave = () => {
    setMousePos(null);
  };

  return (
    <div className="relative w-full aspect-[4/3.4] rounded-2xl overflow-hidden border border-slate-800 bg-[#04070e] shadow-2xl group cursor-crosshair">
      <canvas
        ref={canvasRef}
        width={640}
        height={540}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="w-full h-full object-cover block"
      />
    </div>
  );
};

/**
 * Draws high-fidelity anatomical Brain MRI slice on canvas
 */
function drawAxialBrainMRI(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  mriCase: MRICase
) {
  const cx = w * 0.5;
  const cy = h * 0.5;

  // Background deep black DICOM canvas
  ctx.fillStyle = "#03060c";
  ctx.fillRect(0, 0, w, h);

  // Scalp & Subcutaneous Tissue
  ctx.beginPath();
  ctx.ellipse(cx, cy, 215, 245, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#222a38";
  ctx.fill();

  // Cranium / Skull Bone (Hypointense Cortical Bone)
  ctx.beginPath();
  ctx.ellipse(cx, cy, 206, 236, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#070b13";
  ctx.fill();

  // Diploic Space (Spongy Bone marrow hyperintensity)
  ctx.beginPath();
  ctx.ellipse(cx, cy, 200, 230, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "#404c60";
  ctx.lineWidth = 3.5;
  ctx.stroke();

  // Subdural Space / Dura Mater
  ctx.beginPath();
  ctx.ellipse(cx, cy, 194, 224, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#121927";
  ctx.fill();

  // Cerebral Parenchyma (Gray & White matter baseline)
  ctx.beginPath();
  ctx.ellipse(cx, cy, 190, 220, 0, 0, Math.PI * 2);
  const brainGrad = ctx.createRadialGradient(cx, cy, 40, cx, cy, 190);
  brainGrad.addColorStop(0, "#5a667a"); // White matter (centrally hyperintense)
  brainGrad.addColorStop(0.7, "#3b4554"); // Subcortical white matter
  brainGrad.addColorStop(1, "#28313e"); // Cortical gray matter ribbon
  ctx.fillStyle = brainGrad;
  ctx.fill();

  // Interhemispheric Falx Cerebri Fissure
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(cx, cy - 215);
  ctx.lineTo(cx, cy + 215);
  ctx.strokeStyle = "rgba(10, 15, 25, 0.85)";
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.restore();

  // Cortical Sulcal folds and gyri convolutions
  drawCorticalSulci(ctx, cx, cy);

  // Lateral Ventricles (CSF - Cerebrospinal Fluid hypointensity)
  drawLateralVentricles(ctx, cx, cy, mriCase.massEffect !== "None / Symmetrical");

  // Pathology: Draw Real Lesion if Case is Abnormal
  if (mriCase.classification !== "Healthy") {
    drawTumorPathology(ctx, mriCase);
  }
}

/**
 * Draws anatomical cortical convolutions / sulci
 */
function drawCorticalSulci(ctx: CanvasRenderingContext2D, cx: number, cy: number) {
  ctx.save();
  ctx.strokeStyle = "rgba(18, 24, 38, 0.75)";
  ctx.lineWidth = 2.5;

  const sulci = [
    // Frontal lobes
    { sx: cx - 40, sy: cy - 140, ex: cx - 120, ey: cy - 110 },
    { sx: cx + 40, sy: cy - 140, ex: cx + 120, ey: cy - 110 },
    { sx: cx - 50, sy: cy - 80, ex: cx - 150, ey: cy - 60 },
    { sx: cx + 50, sy: cy - 80, ex: cx + 150, ey: cy - 60 },
    // Sylvian Fissures / Temporal
    { sx: cx - 60, sy: cy + 10, ex: cx - 165, ey: cy + 25 },
    { sx: cx + 60, sy: cy + 10, ex: cx + 165, ey: cy + 25 },
    // Parietal & Occipital
    { sx: cx - 45, sy: cy + 90, ex: cx - 140, ey: cy + 120 },
    { sx: cx + 45, sy: cy + 90, ex: cx + 140, ey: cy + 120 },
    { sx: cx - 30, sy: cy + 150, ex: cx - 90, ey: cy + 180 },
    { sx: cx + 30, sy: cy + 150, ex: cx + 90, ey: cy + 180 },
  ];

  sulci.forEach((s) => {
    ctx.beginPath();
    ctx.moveTo(s.sx, s.sy);
    ctx.quadraticCurveTo(
      (s.sx + s.ex) * 0.5 + (Math.random() * 8 - 4),
      (s.sy + s.ey) * 0.5 + (Math.random() * 8 - 4),
      s.ex,
      s.ey
    );
    ctx.stroke();
  });

  ctx.restore();
}

/**
 * Draws bilateral lateral ventricles with optional mass effect distortion
 */
function drawLateralVentricles(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  hasMassEffect: boolean
) {
  ctx.save();
  ctx.fillStyle = "#05080e"; // Dark CSF hypointensity
  ctx.strokeStyle = "#1a2332";
  ctx.lineWidth = 1.5;

  // Left Ventricle (Anterior horn to body)
  ctx.beginPath();
  ctx.moveTo(cx - 10, cy - 65);
  ctx.bezierCurveTo(cx - 32, cy - 40, cx - 35, cy + 30, cx - 12, cy + 50);
  ctx.bezierCurveTo(cx - 8, cy + 20, cx - 14, cy - 20, cx - 10, cy - 65);
  ctx.fill();
  ctx.stroke();

  // Right Ventricle (Compacted if tumor is on right side causing midline shift)
  const shift = hasMassEffect ? 12 : 0;
  const comp = hasMassEffect ? 0.45 : 1.0;

  ctx.beginPath();
  ctx.moveTo(cx + 10 + shift, cy - 65);
  ctx.bezierCurveTo(
    cx + (32 * comp) + shift,
    cy - 40,
    cx + (35 * comp) + shift,
    cy + 30,
    cx + 12 + shift,
    cy + 50
  );
  ctx.bezierCurveTo(
    cx + 8 + shift,
    cy + 20,
    cx + (14 * comp) + shift,
    cy - 20,
    cx + 10 + shift,
    cy - 65
  );
  ctx.fill();
  ctx.stroke();

  // Third ventricle slit
  ctx.beginPath();
  ctx.ellipse(cx + (hasMassEffect ? 6 : 0), cy + 10, 3, 16, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

/**
 * Draws authentic tumor pathology (necrosis, ring enhancement, edema)
 */
function drawTumorPathology(ctx: CanvasRenderingContext2D, c: MRICase) {
  const { x, y, radius } = c.tumorCentroid;

  ctx.save();

  // 1. Vasogenic Edema Halo (Hyperintense on T2/FLAIR)
  const edemaGrad = ctx.createRadialGradient(x, y, radius * 0.8, x, y, radius * 1.7);
  edemaGrad.addColorStop(0, "rgba(195, 215, 245, 0.45)");
  edemaGrad.addColorStop(0.7, "rgba(140, 170, 215, 0.2)");
  edemaGrad.addColorStop(1, "rgba(80, 100, 140, 0)");
  ctx.fillStyle = edemaGrad;
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.7, 0, Math.PI * 2);
  ctx.fill();

  // 2. Peripheral Ring Enhancement (Hyperintense active rim)
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = "#dce7f7"; // Bright contrast enhancement
  ctx.shadowColor = "#80b0f0";
  ctx.shadowBlur = 10;
  ctx.fill();

  // 3. Central Necrotic Hypointense Core (Hallmark of Glioblastoma)
  if (c.classification === "Glioma") {
    ctx.beginPath();
    ctx.arc(x + 4, y - 3, radius * 0.52, 0, Math.PI * 2);
    ctx.fillStyle = "#1e2430"; // Dark central tissue necrosis
    ctx.shadowBlur = 0;
    ctx.fill();
  }

  // 4. Meningioma Dural Tail
  if (c.classification === "Meningioma") {
    ctx.beginPath();
    ctx.moveTo(x - radius * 0.8, y - radius * 0.6);
    ctx.quadraticCurveTo(x - radius * 1.5, y - radius * 1.2, x - radius * 1.9, y - radius * 1.5);
    ctx.strokeStyle = "#d0e0f8";
    ctx.lineWidth = 4;
    ctx.stroke();
  }

  ctx.restore();
}

/**
 * Renders Explainable AI Grad-CAM Jet/Hot Heatmap
 */
function drawGradCAMHeatmap(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  c: MRICase,
  opacity: number,
  colormode: "jet" | "hot" | "plasma"
) {
  const { x, y, radius } = c.tumorCentroid;
  if (radius <= 0) return;

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.globalCompositeOperation = "screen";

  const heatRadius = radius * 2.3;
  const grad = ctx.createRadialGradient(x, y, radius * 0.2, x, y, heatRadius);

  if (colormode === "jet") {
    // Jet: Deep Blue -> Cyan -> Yellow -> Intense Red
    grad.addColorStop(0, "rgba(255, 10, 10, 0.95)"); // Focal Peak Red
    grad.addColorStop(0.25, "rgba(255, 120, 0, 0.85)"); // Orange
    grad.addColorStop(0.5, "rgba(255, 235, 10, 0.7)"); // Yellow
    grad.addColorStop(0.72, "rgba(10, 220, 180, 0.45)"); // Cyan/Green
    grad.addColorStop(0.9, "rgba(10, 60, 220, 0.2)"); // Blue
    grad.addColorStop(1, "rgba(10, 30, 120, 0)"); // Transparent Blue
  } else if (colormode === "hot") {
    // Hot: White -> Yellow -> Red -> Dark
    grad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
    grad.addColorStop(0.3, "rgba(255, 220, 30, 0.85)");
    grad.addColorStop(0.65, "rgba(230, 40, 20, 0.6)");
    grad.addColorStop(1, "rgba(60, 0, 0, 0)");
  } else {
    // Plasma: Yellow -> Magenta -> Purple -> Dark
    grad.addColorStop(0, "rgba(250, 230, 30, 0.95)");
    grad.addColorStop(0.4, "rgba(210, 50, 140, 0.75)");
    grad.addColorStop(0.75, "rgba(90, 10, 170, 0.4)");
    grad.addColorStop(1, "rgba(20, 0, 80, 0)");
  }

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, heatRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

/**
 * Draws glowing contour boundary & bounding box around the lesion
 */
function drawLesionContour(ctx: CanvasRenderingContext2D, c: MRICase) {
  const { x, y, radius } = c.tumorCentroid;
  if (radius <= 0) return;

  ctx.save();
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 2.2;
  ctx.setLineDash([5, 4]);

  // Bounding Contour
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.15, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Crosshair center
  ctx.beginPath();
  ctx.moveTo(x - 8, y);
  ctx.lineTo(x + 8, y);
  ctx.moveTo(x, y - 8);
  ctx.lineTo(x, y + 8);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Floating Lesion Badge
  const badgeX = x + radius * 1.25;
  const badgeY = y - 10;

  ctx.fillStyle = "rgba(10, 15, 28, 0.92)";
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(badgeX, badgeY - 14, 115, 38, [6]);
  ctx.fill();
  ctx.stroke();

  ctx.font = "bold 11px ui-monospace, monospace";
  ctx.fillStyle = "#f87171";
  ctx.fillText("ROI: TUMOR CORE", badgeX + 8, badgeY + 1);

  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#38bdf8";
  ctx.fillText(`Area: ${c.tumorAreaMm2} mm²`, badgeX + 8, badgeY + 16);

  ctx.restore();
}

/**
 * Draws PACS DICOM Telemetry on Canvas corners
 */
function drawPacsTelemetry(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  c: MRICase,
  pos: { x: number; y: number } | null
) {
  ctx.save();
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillStyle = "rgba(148, 163, 184, 0.85)";

  // Top Left: Patient & Hospital Info
  ctx.fillText("NEUROSCAN PACS v4.2", 16, 22);
  ctx.fillText(`MRN: ${c.mrn}`, 16, 38);
  ctx.fillText(`${c.patientName} (${c.gender}, ${c.age}y)`, 16, 54);

  // Top Right: Sequence & Slicing Info
  ctx.textAlign = "right";
  ctx.fillText(c.scanType, w - 16, 22);
  ctx.fillText("Matrix: 512 x 512 | TE: 90 | TR: 4800", w - 16, 38);
  ctx.fillText("Slice Thickness: 3.0 mm | Axial", w - 16, 54);

  // Bottom Left: Model & Backbone
  ctx.textAlign = "left";
  ctx.fillText("Backbone: DenseNet-121 + Grad-CAM", 16, h - 38);
  ctx.fillText("Layer: conv5_block3_concat (256 Feat)", 16, h - 22);

  // Bottom Right: Real-Time Voxel Coordinates
  ctx.textAlign = "right";
  if (pos) {
    ctx.fillStyle = "#38bdf8";
    ctx.fillText(`Cursor X: ${pos.x} Y: ${pos.y} | Val: 184 HU`, w - 16, h - 22);
  } else {
    ctx.fillText("FOV: 240 mm | Zoom: 1.0x", w - 16, h - 22);
  }

  // Orientation Markers (R / L / A / P)
  ctx.font = "bold 13px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = "rgba(241, 245, 249, 0.55)";
  ctx.textAlign = "center";
  ctx.fillText("A", w * 0.5, 24); // Anterior
  ctx.fillText("P", w * 0.5, h - 14); // Posterior
  ctx.fillText("R", 20, h * 0.5); // Right
  ctx.fillText("L", w - 20, h * 0.5); // Left

  ctx.restore();
}
