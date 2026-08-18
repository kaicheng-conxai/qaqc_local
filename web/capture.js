import { createInterface } from "node:readline";
import { chromium } from "playwright";

const [, , url] = process.argv;
if (!url) throw new Error("Usage: node capture.js <url>");

const readyTimeout = Number(process.env.LOCAL_3D_READY_TIMEOUT_MS ?? "60000");
const itemTimeout = Number(process.env.LOCAL_3D_ITEM_TIMEOUT_MS ?? "60000");
const headed = process.env.MATTERPORT_CAPTURE_HEADED === "1";
let browser;
let page;

const withTimeout = async (promise, ms, label) => {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms); })]);
  } finally {
    clearTimeout(timer);
  }
};

async function closeSession() {
  const current = browser;
  browser = undefined;
  page = undefined;
  if (current) await current.close().catch(() => {});
}

async function startSession() {
  await closeSession();
  browser = await chromium.launch({ headless: !headed });
  page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on("console", (message) => { if (message.type() === "error") process.stderr.write(`[browser:error] ${message.text()}\n`); });
  page.on("pageerror", (error) => process.stderr.write(`[browser:error] ${error.message}\n`));
  process.stderr.write(`Loading Matterport Bundle: headed=${headed} readyTimeoutMs=${readyTimeout}\n`);
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await withTimeout(page.evaluate(() => window.backfillReady), readyTimeout, "Matterport SDK readiness");
  process.stderr.write("Matterport Bundle ready\n");
}

async function ensureSession() {
  if (!browser?.isConnected() || page?.isClosed()) await startSession();
}

async function capture(request) {
  await ensureSession();
  for (const item of request.items ?? []) {
    Object.assign(item, await withTimeout(
      page.evaluate(({ item: current, settleMs }) => window.captureAnchor(current, settleMs), { item, settleMs: Number(request.settleMs ?? 120) }),
      itemTimeout,
      `Matterport capture ${item.workItemId}`,
    ));
  }
  return request.items ?? [];
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
  if (!line.trim()) continue;
  const request = JSON.parse(line);
  if (request.type === "shutdown") break;
  try {
    const items = await capture(request);
    process.stdout.write(`${JSON.stringify({ id: request.id, items })}\n`);
  } catch (error) {
    await closeSession();
    process.stdout.write(`${JSON.stringify({ id: request.id, error: String(error?.message ?? error) })}\n`);
  }
}
await closeSession();
