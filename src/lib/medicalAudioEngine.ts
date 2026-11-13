"use client";

class MedicalAudioEngine {
  private ctx: AudioContext | null = null;
  private isMuted: boolean = false;
  private voiceEnabled: boolean = true;
  private lastSpokenTime: number = 0;

  private initContext() {
    if (typeof window === "undefined") return null;
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
    return this.ctx;
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted;
  }

  public setVoiceEnabled(enabled: boolean) {
    this.voiceEnabled = enabled;
  }

  public getMuted() {
    return this.isMuted;
  }

  public getVoiceEnabled() {
    return this.voiceEnabled;
  }

  /**
   * CRITICAL MALIGNANT TUMOR ALARM (RED ALERT):
   * Rapid clinical telemetry warning pulses (880Hz square wave with sharp attack/decay)
   */
  public playMalignantAlarm() {
    if (this.isMuted) return;
    const ctx = this.initContext();
    if (!ctx) return;

    const startTime = ctx.currentTime;
    const pulseCount = 3;
    const duration = 0.09;
    const gap = 0.06;

    for (let i = 0; i < pulseCount; i++) {
      const t = startTime + i * (duration + gap);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "square";
      osc.frequency.setValueAtTime(880, t); // A5 alert
      osc.frequency.setValueAtTime(988, t + duration * 0.5); // B5 dissonance

      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(0.3, t + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, t + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(t);
      osc.stop(t + duration);
    }
  }

  /**
   * CAUTION / LOW-GRADE LESION PING (AMBER ALERT):
   * Diagnostic dual-frequency warning tone
   */
  public playCautionAlert() {
    if (this.isMuted) return;
    const ctx = this.initContext();
    if (!ctx) return;

    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "triangle";
    osc.frequency.setValueAtTime(587.33, t); // D5
    osc.frequency.exponentialRampToValueAtTime(440, t + 0.28); // A4

    gain.gain.setValueAtTime(0.25, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(t);
    osc.stop(t + 0.3);
  }

  /**
   * HEALTHY / NEGATIVE SCAN CONFIRMATION (GREEN ALERT):
   * Harmonic C-Major medical confirmation chime (523Hz -> 659Hz -> 784Hz)
   */
  public playHealthyChime() {
    if (this.isMuted) return;
    const ctx = this.initContext();
    if (!ctx) return;

    const frequencies = [523.25, 659.25, 783.99]; // C5, E5, G5
    const startTime = ctx.currentTime;

    frequencies.forEach((freq, idx) => {
      const t = startTime + idx * 0.09;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, t);

      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(0.28, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.4);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(t);
      osc.stop(t + 0.4);
    });
  }

  /**
   * RADIOLOGY SPOKEN VOICE BRIEFING:
   * Professional medical speech synthesis
   */
  public speakRadiologyReport(text: string) {
    if (!this.voiceEnabled || typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    const now = Date.now();
    if (now - this.lastSpokenTime < 2500) {
      window.speechSynthesis.cancel();
    }
    this.lastSpokenTime = now;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05; // Clear medical cadence
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    window.speechSynthesis.speak(utterance);
  }
}

export const medicalAudio = new MedicalAudioEngine();
