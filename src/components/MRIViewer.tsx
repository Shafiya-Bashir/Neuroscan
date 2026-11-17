"use client";

import React, { useEffect, useRef, useState } from "react";
import { MRICase } from "@/lib/mriData";

export type VisualizationMode = "fused" | "difference" | "spotlight" | "split";

interface Props {
  currentCase: MRICase;
  gradCamOpacity: number; // 0 to 1
  showContour: boolean;
  showHeatmap: boolean;
  colorMap: "jet" | "hot" | "plasma";
  visualizationMode: VisualizationMode;
  splitPosition: number; // 0 to 1
  onSplitChange?: (pos: number) => void;
}

export const MRIViewer: React.FC<Props> = ({
  currentCase,
  gradCamOpacity,
  showContour,
  showHeatmap,
  colorMap,
  visualizationMode,
  splitPosition,
  onSplitChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [isDraggingWiper, setIsDraggingWiper] = useState<boolean>(false);
  const [cachedImg, setCachedImg] = useState<HTMLImageElement | null>(null);

  // Preload and cache custom/uploaded MRI scan if provided
  useEffect(() => {
    if (currentCase.customImageUrl) {
      const img = new Image();
      img.src = currentCase.customImageUrl;
      img.onload = () => {
        setCachedImg(img);
      };
      img.onerror = () => {
        setCachedImg(null);
      };
    } else {
      setCachedImg(null);
    }
  }, [currentCase.customImageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (visualizationMode === "split") {
      // Split Screen Wiper: Left is Raw MRI, Right is Tumor Detection / Grad-CAM / Difference
      const splitX = Math.round(width * splitPosition);

      // 1. Draw Raw MRI on Left side
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, height);
      ctx.clip();
      drawAxialBrainMRI(ctx, width, height, currentCase, false, cachedImg);
      drawPacsTelemetry(ctx, width, height, currentCase, mousePos, "RAW MRI (UNALTERED)");
      ctx.restore();

      // 2. Draw Fused Detection & Grad-CAM on Right side
      ctx.save();
      ctx.beginPath();
      ctx.rect(splitX, 0, width - splitX, height);
      ctx.clip();
      drawAxialBrainMRI(ctx, width, height, currentCase, true, cachedImg);
      if (currentCase.classification !== "Healthy") {
        drawGradCAMHeatmap(ctx, width, height, currentCase, gradCamOpacity, colorMap);
        drawDifferenceOverlay(ctx, currentCase);
        if (showContour) {
          drawLesionContour(ctx, currentCase);
        }
      }
      drawPacsTelemetry(ctx, width, height, currentCase, mousePos, "AI DETECTION & DIFFERENCE MAP");
      ctx.restore();

      // 3. Draw Wiper Dividing Line & Handle
      drawWiperLine(ctx, splitX, height);
    } else {
      // Single Viewport Modes
      // 1. Draw Axial MRI
      const drawLesion = currentCase.classification !== "Healthy";
      drawAxialBrainMRI(ctx, width, height, currentCase, drawLesion, cachedImg);

      // 2. Mode-Specific Layers
      if (currentCase.classification !== "Healthy") {
        if (visualizationMode === "difference") {
          // Highlight strictly the subtraction / difference between normal and tumor
          drawDifferenceIsolationMode(ctx, width, height, currentCase);
        } else if (visualizationMode === "spotlight") {
          // Spotlight strictly the detected portion of the brain where tumor is located
          drawSpotlightDetectedPortion(ctx, width, height, currentCase);
        } else {
          // Default Fused Grad-CAM
          if (showHeatmap && gradCamOpacity > 0) {
            drawGradCAMHeatmap(ctx, width, height, currentCase, gradCamOpacity, colorMap);
          }
        }

        // Lesion Contour & Reticle
        if (showContour) {
          drawLesionContour(ctx, currentCase);
        }
      }

      // Telemetry & Anatomical Landmark Callout
      drawPacsTelemetry(
        ctx,
        width,
        height,
        currentCase,
        mousePos,
        visualizationMode === "difference"
          ? "SUBTRACTION DIFFERENCE VIEW"
          : visualizationMode === "spotlight"
          ? "DETECTED PORTION SPOTLIGHT"
          : "FUSED GRAD-CAM OVERLAY"
      );

      if (currentCase.classification !== "Healthy") {
        drawDetectedPortionHUD(ctx, width, height, currentCase);
      }
    }
  }, [
    currentCase,
    cachedImg,
    gradCamOpacity,
    showContour,
    showHeatmap,
    colorMap,
    visualizationMode,
    splitPosition,
    mousePos,
  ]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 640);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 540);
    setMousePos({ x, y });

    if (isDraggingWiper && onSplitChange) {
      const relX = Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width));
      onSplitChange(relX);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (visualizationMode === "split") {
      setIsDraggingWiper(true);
      if (onSplitChange) {
        const rect = e.currentTarget.getBoundingClientRect();
        const relX = Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width));
        onSplitChange(relX);
      }
    }
  };

  const handleMouseUp = () => {
    setIsDraggingWiper(false);
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-[4/3.4] rounded-2xl overflow-hidden border border-slate-800 bg-[#04070e] shadow-2xl group cursor-crosshair select-none"
    >
      <canvas
        ref={canvasRef}
        width={640}
        height={540}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setMousePos(null);
          setIsDraggingWiper(false);
        }}
        className="w-full h-full object-cover block"
      />
    </div>
  );
};

/**
 * Draws either the authentic uploaded image or the anatomical procedural scan
 */
function drawAxialBrainMRI(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  mriCase: MRICase,
  includeLesion: boolean,
  cachedImg: HTMLImageElement | null
) {
  const cx = w * 0.5;
  const cy = h * 0.5;

  // Background DICOM canvas
  ctx.fillStyle = "#03060c";
  ctx.fillRect(0, 0, w, h);

  if (cachedImg) {
    // Render the real uploaded or demo image
    const imgW = cachedImg.width || 256;
    const imgH = cachedImg.height || 256;
    const scale = Math.min((w * 0.86) / imgW, (h * 0.86) / imgH);
    const dw = imgW * scale;
    const dh = imgH * scale;
    const dx = (w - dw) / 2;
    const dy = (h - dh) / 2;

    ctx.save();
    ctx.drawImage(cachedImg, dx, dy, dw, dh);
    ctx.restore();
    return;
  }

  // Scalp & Subcutaneous Tissue
  ctx.beginPath();
  ctx.ellipse(cx, cy, 215, 245, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#222a38";
  ctx.fill();

  // Cranium / Skull Bone
  ctx.beginPath();
  ctx.ellipse(cx, cy, 206, 236, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#070b13";
  ctx.fill();

  // Diploic Space
  ctx.beginPath();
  ctx.ellipse(cx, cy, 200, 230, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "#404c60";
  ctx.lineWidth = 3.5;
  ctx.stroke();

  // Dura Mater
  ctx.beginPath();
  ctx.ellipse(cx, cy, 194, 224, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#121927";
  ctx.fill();

  // Cerebral Parenchyma
  ctx.beginPath();
  ctx.ellipse(cx, cy, 190, 220, 0, 0, Math.PI * 2);
  const brainGrad = ctx.createRadialGradient(cx, cy, 40, cx, cy, 190);
  brainGrad.addColorStop(0, "#5a667a");
  brainGrad.addColorStop(0.7, "#3b4554");
  brainGrad.addColorStop(1, "#28313e");
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

  // Lateral Ventricles
  drawLateralVentricles(ctx, cx, cy, mriCase.massEffect !== "None / Symmetrical");

  // Draw authentic tumor pathology if requested
  if (includeLesion && mriCase.classification !== "Healthy") {
    drawTumorPathology(ctx, mriCase);
  }
}

/**
 * Draws anatomical cortical sulci
 */
function drawCorticalSulci(ctx: CanvasRenderingContext2D, cx: number, cy: number) {
  ctx.save();
  ctx.strokeStyle = "rgba(18, 24, 38, 0.75)";
  ctx.lineWidth = 2.5;

  const sulci = [
    { sx: cx - 40, sy: cy - 140, ex: cx - 120, ey: cy - 110 },
    { sx: cx + 40, sy: cy - 140, ex: cx + 120, ey: cy - 110 },
    { sx: cx - 50, sy: cy - 80, ex: cx - 150, ey: cy - 60 },
    { sx: cx + 50, sy: cy - 80, ex: cx + 150, ey: cy - 60 },
    { sx: cx - 60, sy: cy + 10, ex: cx - 165, ey: cy + 25 },
    { sx: cx + 60, sy: cy + 10, ex: cx + 165, ey: cy + 25 },
    { sx: cx - 45, sy: cy + 90, ex: cx - 140, ey: cy + 120 },
    { sx: cx + 45, sy: cy + 90, ex: cx + 140, ey: cy + 120 },
    { sx: cx - 30, sy: cy + 150, ex: cx - 90, ey: cy + 180 },
    { sx: cx + 30, sy: cy + 150, ex: cx + 90, ey: cy + 180 },
  ];

  sulci.forEach((s) => {
    ctx.beginPath();
    ctx.moveTo(s.sx, s.sy);
    ctx.quadraticCurveTo(
      (s.sx + s.ex) * 0.5 + 2,
      (s.sy + s.ey) * 0.5 - 2,
      s.ex,
      s.ey
    );
    ctx.stroke();
  });

  ctx.restore();
}

/**
 * Draws bilateral lateral ventricles
 */
function drawLateralVentricles(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  hasMassEffect: boolean
) {
  ctx.save();
  ctx.fillStyle = "#05080e";
  ctx.strokeStyle = "#1a2332";
  ctx.lineWidth = 1.5;

  // Left Ventricle
  ctx.beginPath();
  ctx.moveTo(cx - 10, cy - 65);
  ctx.bezierCurveTo(cx - 32, cy - 40, cx - 35, cy + 30, cx - 12, cy + 50);
  ctx.bezierCurveTo(cx - 8, cy + 20, cx - 14, cy - 20, cx - 10, cy - 65);
  ctx.fill();
  ctx.stroke();

  // Right Ventricle
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

  // Third ventricle
  ctx.beginPath();
  ctx.ellipse(cx + (hasMassEffect ? 6 : 0), cy + 10, 3, 16, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

/**
 * Draws authentic tumor pathology
 */
function drawTumorPathology(ctx: CanvasRenderingContext2D, c: MRICase) {
  const { x, y, radius } = c.tumorCentroid;
  if (radius <= 0) return;

  ctx.save();

  // 1. Vasogenic Edema Halo
  const edemaGrad = ctx.createRadialGradient(x, y, radius * 0.8, x, y, radius * 1.7);
  edemaGrad.addColorStop(0, "rgba(195, 215, 245, 0.45)");
  edemaGrad.addColorStop(0.7, "rgba(140, 170, 215, 0.2)");
  edemaGrad.addColorStop(1, "rgba(80, 100, 140, 0)");
  ctx.fillStyle = edemaGrad;
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.7, 0, Math.PI * 2);
  ctx.fill();

  // 2. Peripheral Ring Enhancement
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = "#dce7f7";
  ctx.shadowColor = "#80b0f0";
  ctx.shadowBlur = 10;
  ctx.fill();

  // 3. Central Necrotic Hypointense Core (Glioma)
  if (c.classification === "Glioma") {
    ctx.beginPath();
    ctx.arc(x + 4, y - 3, radius * 0.52, 0, Math.PI * 2);
    ctx.fillStyle = "#1e2430";
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
 * Renders Grad-CAM Jet/Hot/Plasma Heatmap
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
    grad.addColorStop(0, "rgba(255, 10, 10, 0.95)");
    grad.addColorStop(0.25, "rgba(255, 120, 0, 0.85)");
    grad.addColorStop(0.5, "rgba(255, 235, 10, 0.7)");
    grad.addColorStop(0.72, "rgba(10, 220, 180, 0.45)");
    grad.addColorStop(0.9, "rgba(10, 60, 220, 0.2)");
    grad.addColorStop(1, "rgba(10, 30, 120, 0)");
  } else if (colormode === "hot") {
    grad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
    grad.addColorStop(0.3, "rgba(255, 220, 30, 0.85)");
    grad.addColorStop(0.65, "rgba(230, 40, 20, 0.6)");
    grad.addColorStop(1, "rgba(60, 0, 0, 0)");
  } else {
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
 * Mode 2: Subtraction Difference Map (Neon Delta Highlight)
 */
function drawDifferenceIsolationMode(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  c: MRICase
) {
  const { x, y, radius } = c.tumorCentroid;
  if (radius <= 0) return;

  ctx.save();
  // Dark wash over background
  ctx.fillStyle = "rgba(4, 7, 14, 0.72)";
  ctx.fillRect(0, 0, w, h);

  // Vivid Subtraction Delta Glow (Neon Emerald/Cyan)
  const diffGrad = ctx.createRadialGradient(x, y, radius * 0.2, x, y, radius * 1.6);
  diffGrad.addColorStop(0, "rgba(52, 211, 153, 0.95)");
  diffGrad.addColorStop(0.5, "rgba(6, 182, 212, 0.7)");
  diffGrad.addColorStop(0.85, "rgba(14, 165, 233, 0.35)");
  diffGrad.addColorStop(1, "rgba(0, 0, 0, 0)");

  ctx.globalCompositeOperation = "lighter";
  ctx.fillStyle = diffGrad;
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.6, 0, Math.PI * 2);
  ctx.fill();

  // Concentric Delta Rings
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.1, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.4, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

/**
 * Mode 3: Spotlight Detected Tumor Portion
 */
function drawSpotlightDetectedPortion(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  c: MRICase
) {
  const { x, y, radius } = c.tumorCentroid;
  if (radius <= 0) return;

  ctx.save();
  const spotGrad = ctx.createRadialGradient(x, y, radius * 0.8, x, y, radius * 2.8);
  spotGrad.addColorStop(0, "rgba(0, 0, 0, 0)");
  spotGrad.addColorStop(0.6, "rgba(3, 7, 18, 0.65)");
  spotGrad.addColorStop(1, "rgba(3, 7, 18, 0.92)");

  ctx.fillStyle = spotGrad;
  ctx.fillRect(0, 0, w, h);

  // Pulse targeting box around detected portion
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  const boxPad = radius * 1.35;
  ctx.strokeRect(x - boxPad, y - boxPad, boxPad * 2, boxPad * 2);

  // Targeting Corner Accents
  const cornerLen = 14;
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3.5;

  // Top-left
  ctx.beginPath();
  ctx.moveTo(x - boxPad, y - boxPad + cornerLen);
  ctx.lineTo(x - boxPad, y - boxPad);
  ctx.lineTo(x - boxPad + cornerLen, y - boxPad);
  ctx.stroke();

  // Top-right
  ctx.beginPath();
  ctx.moveTo(x + boxPad - cornerLen, y - boxPad);
  ctx.lineTo(x + boxPad, y - boxPad);
  ctx.lineTo(x + boxPad, y - boxPad + cornerLen);
  ctx.stroke();

  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(x - boxPad, y + boxPad - cornerLen);
  ctx.lineTo(x - boxPad, y + boxPad);
  ctx.lineTo(x - boxPad + cornerLen, y + boxPad);
  ctx.stroke();

  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(x + boxPad - cornerLen, y + boxPad);
  ctx.lineTo(x + boxPad, y + boxPad);
  ctx.lineTo(x + boxPad, y + boxPad - cornerLen);
  ctx.stroke();

  ctx.restore();
}

/**
 * Difference Overlay for Split Mode
 */
function drawDifferenceOverlay(ctx: CanvasRenderingContext2D, c: MRICase) {
  const { x, y, radius } = c.tumorCentroid;
  if (radius <= 0) return;

  ctx.save();
  ctx.globalAlpha = 0.5;
  ctx.fillStyle = "rgba(16, 185, 129, 0.4)";
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.25, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * Interactive Wiper Line in Split Mode
 */
function drawWiperLine(ctx: CanvasRenderingContext2D, splitX: number, h: number) {
  ctx.save();
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(splitX, 0);
  ctx.lineTo(splitX, h);
  ctx.stroke();

  const knobY = h * 0.5;
  ctx.fillStyle = "#0284c7";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(splitX, knobY, 18, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 13px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("⮂ ⮃", splitX, knobY);

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
  const badgeX = Math.min(x + radius * 1.25, 510);
  const badgeY = Math.max(y - 10, 40);

  ctx.fillStyle = "rgba(10, 15, 28, 0.92)";
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(badgeX, badgeY - 14, 120, 42, [6]);
  ctx.fill();
  ctx.stroke();

  ctx.font = "bold 11px ui-monospace, monospace";
  ctx.fillStyle = "#f87171";
  ctx.fillText("ROI: DETECTED TUMOR", badgeX + 8, badgeY + 1);

  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#38bdf8";
  ctx.fillText(`Area: ${c.tumorAreaMm2} mm²`, badgeX + 8, badgeY + 16);
  ctx.fillText(`(X:${Math.round(x)}, Y:${Math.round(y)})`, badgeX + 8, badgeY + 28);

  ctx.restore();
}

/**
 * Detected Portion HUD Callout
 */
function drawDetectedPortionHUD(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  c: MRICase
) {
  ctx.save();
  const bannerY = h - 64;

  ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(16, bannerY, w - 32, 42, [8]);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#38bdf8";
  ctx.font = "bold 11px ui-monospace, monospace";
  ctx.fillText("🎯 DETECTED LESION PORTION:", 28, bannerY + 17);

  ctx.font = "bold 12px ui-sans-serif, system-ui";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(c.brainRegion, 205, bannerY + 17);

  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(
    `Centroid [X: ${c.tumorCentroid.x}, Y: ${c.tumorCentroid.y}] | Radius: ${c.tumorCentroid.radius}mm | Area: ${c.tumorAreaMm2} mm² | Mass Effect: ${c.massEffect}`,
    28,
    bannerY + 33
  );

  ctx.restore();
}

/**
 * Draws PACS DICOM Telemetry
 */
function drawPacsTelemetry(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  c: MRICase,
  pos: { x: number; y: number } | null,
  viewModeLabel: string
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
  ctx.fillText("Matrix: 256 x 256 | TE: 90 | TR: 4800", w - 16, 38);
  ctx.fillText(`MODE: ${viewModeLabel}`, w - 16, 54);

  // Orientation Markers
  ctx.font = "bold 13px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = "rgba(241, 245, 249, 0.55)";
  ctx.textAlign = "center";
  ctx.fillText("A", w * 0.5, 24);
  ctx.fillText("P", w * 0.5, h - 8);
  ctx.fillText("R", 16, h * 0.5);
  ctx.fillText("L", w - 16, h * 0.5);

  ctx.restore();
}
