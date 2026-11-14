// Forecast Polymarket FAA outage attribution prices, 240 minutes ahead.
// Model: per-market random walk, no drift. Median stays at the checkpoint
// price x. Spread from a robust scale of the market's 1-minute moves
// (MAD of diffs), clamped to [0.0002, 0.008], with a jump guard that
// widens the spread after a large recent move. Deterministic, stdlib only.
import * as fs from "node:fs";
import * as path from "node:path";

const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
const SIGMA_FLOOR = 0.0002;
const SIGMA_CAP = 0.008;
const HORIZON = 240;

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

// Robust per-minute volatility from 1-minute price changes.
// Diffs are mostly zero (prices move in steps), so the plain MAD of all
// diffs often collapses to zero. Use the MAD of the nonzero moves and
// scale by sqrt(move frequency): if moves of typical size s happen with
// probability p per minute, the per-minute std is about s * sqrt(p).
// Only the most recent 24h of moves are used: volatility regimes drift
// over 72h and the checkpoint's neighbourhood is what the next 4h face.
// A 1.15 calibration factor widens the spread slightly: pinball loss
// punishes a too-narrow forecast after jumps more than a slightly wide
// one costs at quiet minutes.
const SIGMA_CALIBRATION = 1.15;
const SIGMA_WINDOW_MINUTES = 1440;

function estimateSigma(prices: number[]): number {
  if (prices.length < 2) return SIGMA_FLOOR;
  const start = Math.max(1, prices.length - SIGMA_WINDOW_MINUTES);
  const diffs: number[] = [];
  for (let i = start; i < prices.length; i++) {
    diffs.push(prices[i] - prices[i - 1]);
  }
  const absDiffs = diffs.map((d) => Math.abs(d));
  const medAbs = median(absDiffs);
  let sigma: number;
  if (medAbs > 0) {
    // Classic MAD scale: 1.4826 * median(|diff - median(diff)|).
    const medDiff = median(diffs);
    const mad = median(diffs.map((d) => Math.abs(d - medDiff)));
    sigma = 1.4826 * mad;
    if (sigma <= 0) sigma = 1.4826 * medAbs;
  } else {
    const nonzero = absDiffs.filter((d) => d > 0);
    if (nonzero.length === 0) return SIGMA_FLOOR;
    const typicalMove = median(nonzero);
    const p = nonzero.length / absDiffs.length;
    sigma = typicalMove * Math.sqrt(p);
  }
  if (!isFinite(sigma) || sigma <= 0) return SIGMA_FLOOR;
  return Math.min(
    SIGMA_CAP,
    Math.max(SIGMA_FLOOR, sigma * SIGMA_CALIBRATION),
  );
}

// Jump guard: if a large single-minute move happened recently, the
// market is in a jump regime and the near-term spread must be wider.
// Returns a per-minute multiplier that decays back to 1 over ~1 hour.
function jumpMultiplier(prices: number[], sigma: number): number[] {
  const mult: number[] = new Array(HORIZON).fill(1);
  const window = Math.min(15, prices.length - 1);
  let recentMax = 0;
  for (let i = prices.length - window; i < prices.length; i++) {
    const d = Math.abs(prices[i] - prices[i - 1]);
    if (d > recentMax) recentMax = d;
  }
  const threshold = Math.max(0.02, 8 * sigma);
  if (recentMax <= threshold) return mult;
  // Strength grows with how far the jump exceeds the threshold.
  const strength = Math.min(2.5, 1 + (recentMax - threshold) / Math.max(threshold, 0.01));
  for (let h = 1; h <= HORIZON; h++) {
    mult[h - 1] = 1 + (strength - 1) * Math.exp(-h / 45);
  }
  return mult;
}

function clamp01(v: number): number {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function round6(v: number): number {
  return Math.round(v * 1e6) / 1e6;
}

function forecastMarket(prices: number[]): number[][] {
  const x = prices[prices.length - 1];
  const sigma = estimateSigma(prices);
  const mult = jumpMultiplier(prices, sigma);
  const rows: number[][] = [];
  for (let h = 1; h <= HORIZON; h++) {
    const s = sigma * mult[h - 1] * Math.sqrt(h);
    const row = Z.map((z) => clamp01(round6(x + z * s)));
    // Enforce non-decreasing within the row after rounding/clamping.
    for (let k = 1; k < row.length; k++) {
      if (row[k] < row[k - 1]) row[k] = row[k - 1];
    }
    rows.push(row);
  }
  return rows;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const caseData = JSON.parse(
      fs.readFileSync(path.join(inputDir, file), "utf8"),
    );
    const forecasts = (caseData.markets as any[]).map((m) => ({
      id: m.id,
      q: forecastMarket(m.prices as number[]),
    }));
    const outId = file.replace(/\.in$/, "");
    fs.writeFileSync(
      path.join(outputDir, `${outId}.out`),
      JSON.stringify({ forecasts }),
    );
  }
}

main();
