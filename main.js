const coverInput = document.getElementById('coverInput');
const stegoInput = document.getElementById('stegoInput');
const coverPreview = document.getElementById('coverPreview');
const stegoPreview = document.getElementById('stegoPreview');
const messageInput = document.getElementById('message');
const seedInput = document.getElementById('seed');
const strengthInput = document.getElementById('strength');
const logBox = document.getElementById('log');
const embedBtn = document.getElementById('embedBtn');
const downloadBtn = document.getElementById('downloadBtn');
const detectBtn = document.getElementById('detectBtn');
const detectResult = document.getElementById('detectResult');
const debugOutput = document.getElementById('debugOutput');
const detectSeedInput = document.getElementById('detectSeed');
const detectLenInput = document.getElementById('detectLen');

let coverImageData = null;
let stegoImageData = null;

function log(message) {
  const time = new Date().toLocaleTimeString();
  logBox.textContent = `[${time}] ${message}\n${logBox.textContent}`.slice(0, 4000);
}

function clamp(v) {
  return Math.min(255, Math.max(0, v));
}

function xmur3(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function() {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return (h ^= h >>> 16) >>> 0;
  };
}

function mulberry32(a) {
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function seededRandom(seedStr) {
  const seedGen = xmur3(seedStr);
  return mulberry32(seedGen());
}

function textToBits(text) {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  const length = bytes.length;
  const bits = [];
  // 16-bit length header
  for (let i = 15; i >= 0; i--) {
    bits.push((length >> i) & 1);
  }
  for (const byte of bytes) {
    for (let i = 7; i >= 0; i--) bits.push((byte >> i) & 1);
  }
  return bits;
}

function bitsToText(bits, forcedLen) {
  if (bits.length < 16) return '';
  let length = 0;
  for (let i = 0; i < 16; i++) {
    length = (length << 1) | bits[i];
  }
  if (Number.isInteger(forcedLen) && forcedLen > 0) length = forcedLen;
  const totalBits = length * 8;
  const outBytes = [];
  for (let i = 0; i < totalBits && 16 + i < bits.length; i += 8) {
    let val = 0;
    for (let j = 0; j < 8; j++) {
      val = (val << 1) | (bits[16 + i + j] || 0);
    }
    outBytes.push(val);
  }
  try {
    return new TextDecoder().decode(new Uint8Array(outBytes));
  } catch (e) {
    return '';
  }
}

function isLikelyGibberish(text) {
  if (!text) return false;
  const trimmed = text.trim();
  if (!trimmed) return false;
  const chars = [...trimmed];
  let noisy = 0;
  for (const ch of chars) {
    const code = ch.codePointAt(0);
    const isCommon = /[\w\s.,;:!?"'`~\-\(\)\[\]\{\}…，。！？【】（）《》、“”‘’·\u4e00-\u9fa5]/u.test(ch);
    const isControl = code !== undefined && (code < 32 || (code >= 0x7f && code <= 0x9f));
    if (!isCommon || isControl || ch === '\ufffd') noisy++;
  }
  const noiseRatio = noisy / chars.length;
  return noiseRatio >= 0.35 || chars.length >= 6 && new Set(chars).size <= 2;
}

function loadImageToCanvas(file, canvas) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        resolve(ctx.getImageData(0, 0, canvas.width, canvas.height));
      };
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function imageDataCopy(src) {
  return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
}

function toLuma(imageData) {
  const { data, width, height } = imageData;
  const luma = new Array(height).fill(0).map(() => new Array(width).fill(0));
  let idx = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      luma[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
      idx += 4;
    }
  }
  return luma;
}

function copyMatrix(matrix) {
  return matrix.map(row => row.slice());
}

function applyLumaToImageData(imageData, luma, baseLuma = null) {
  const { data, width, height } = imageData;
  let idx = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (baseLuma) {
        const delta = clamp(luma[y][x]) - baseLuma[y][x];
        data[idx] = clamp(data[idx] + delta);
        data[idx + 1] = clamp(data[idx + 1] + delta);
        data[idx + 2] = clamp(data[idx + 2] + delta);
      } else {
        const value = clamp(luma[y][x]);
        data[idx] = value;
        data[idx + 1] = value;
        data[idx + 2] = value;
      }
      data[idx + 3] = 255;
      idx += 4;
    }
  }
  return imageData;
}

function dct2(block) {
  const N = 8;
  const coeff = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let sum = 0;
      for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
          sum += block[x][y] * Math.cos(((2 * x + 1) * u * Math.PI) / 16) * Math.cos(((2 * y + 1) * v * Math.PI) / 16);
        }
      }
      const cu = u === 0 ? 1 / Math.sqrt(2) : 1;
      const cv = v === 0 ? 1 / Math.sqrt(2) : 1;
      coeff[u][v] = 0.25 * cu * cv * sum;
    }
  }
  return coeff;
}

function idct2(coeff) {
  const N = 8;
  const block = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      let sum = 0;
      for (let u = 0; u < N; u++) {
        for (let v = 0; v < N; v++) {
          const cu = u === 0 ? 1 / Math.sqrt(2) : 1;
          const cv = v === 0 ? 1 / Math.sqrt(2) : 1;
          sum += cu * cv * coeff[u][v] * Math.cos(((2 * x + 1) * u * Math.PI) / 16) * Math.cos(((2 * y + 1) * v * Math.PI) / 16);
        }
      }
      block[x][y] = 0.25 * sum;
    }
  }
  return block;
}

function embedDCT(imageData, bits, strength = 6) {
  const baseLuma = toLuma(imageData);
  const luma = copyMatrix(baseLuma);
  const w = imageData.width, h = imageData.height;
  const delta = 2 + strength * 0.8;
  const posA = [2, 3], posB = [3, 2];
  let bitIdx = 0;
  for (let by = 0; by < h; by += 8) {
    for (let bx = 0; bx < w; bx += 8) {
      if (bitIdx >= bits.length) break;
      const block = Array.from({ length: 8 }, (_, i) => luma[by + i]?.slice(bx, bx + 8) || new Array(8).fill(0));
      const coeff = dct2(block);
      const bit = bits[bitIdx];
      const diff = coeff[posA[0]][posA[1]] - coeff[posB[0]][posB[1]];
      if (bit === 1 && diff < delta) {
        coeff[posA[0]][posA[1]] += delta - diff;
      } else if (bit === 0 && diff > -delta) {
        coeff[posB[0]][posB[1]] += diff + delta;
      }
      const newBlock = idct2(coeff);
      for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
          if (by + i < h && bx + j < w) luma[by + i][bx + j] = newBlock[i][j];
        }
      }
      bitIdx++;
    }
    if (bitIdx >= bits.length) break;
  }
  log(`DCT 水印写入 ${bitIdx} bits`);
  return applyLumaToImageData(imageData, luma, baseLuma);
}

function extractDCT(imageData, bitCount = 0) {
  const luma = toLuma(imageData);
  const w = imageData.width, h = imageData.height;
  const posA = [2, 3], posB = [3, 2];
  const bits = [];
  for (let by = 0; by < h; by += 8) {
    for (let bx = 0; bx < w; bx += 8) {
      const block = Array.from({ length: 8 }, (_, i) => luma[by + i]?.slice(bx, bx + 8) || new Array(8).fill(0));
      const coeff = dct2(block);
      bits.push(coeff[posA[0]][posA[1]] - coeff[posB[0]][posB[1]] > 0 ? 1 : 0);
      if (bitCount && bits.length >= bitCount) return bits;
    }
  }
  return bits;
}

function haarDwt(matrix) {
  const h = matrix.length, w = matrix[0].length;
  const temp = Array.from({ length: h }, () => new Array(w).fill(0));
  // horizontal
  for (let y = 0; y < h; y++) {
    let k = 0;
    for (let x = 0; x < w; x += 2) {
      temp[y][k] = (matrix[y][x] + matrix[y][x + 1]) / 2;
      temp[y][k + w / 2] = (matrix[y][x] - matrix[y][x + 1]) / 2;
      k++;
    }
  }
  const output = Array.from({ length: h }, () => new Array(w).fill(0));
  // vertical
  for (let x = 0; x < w; x++) {
    let k = 0;
    for (let y = 0; y < h; y += 2) {
      output[k][x] = (temp[y][x] + temp[y + 1][x]) / 2;
      output[k + h / 2][x] = (temp[y][x] - temp[y + 1][x]) / 2;
      k++;
    }
  }
  return output;
}

function haarIdwt(coeff) {
  const h = coeff.length, w = coeff[0].length;
  const temp = Array.from({ length: h }, () => new Array(w).fill(0));
  for (let x = 0; x < w; x++) {
    let k = 0;
    for (let y = 0; y < h / 2; y++) {
      temp[k][x] = coeff[y][x] + coeff[y + h / 2][x];
      temp[k + 1][x] = coeff[y][x] - coeff[y + h / 2][x];
      k += 2;
    }
  }
  const output = Array.from({ length: h }, () => new Array(w).fill(0));
  for (let y = 0; y < h; y++) {
    let k = 0;
    for (let x = 0; x < w / 2; x++) {
      output[y][k] = temp[y][x] + temp[y][x + w / 2];
      output[y][k + 1] = temp[y][x] - temp[y][x + w / 2];
      k += 2;
    }
  }
  return output;
}

function embedDWT(imageData, bits, strength = 6) {
  const baseLuma = toLuma(imageData);
  const luma = copyMatrix(baseLuma);
  const hEven = luma.length - (luma.length % 2);
  const wEven = luma[0].length - (luma[0].length % 2);
  const trimmed = luma.slice(0, hEven).map(row => row.slice(0, wEven));
  const coeff = haarDwt(trimmed);
  const delta = 0.75 * strength;
  let bitIdx = 0;
  for (let y = hEven / 2; y < hEven && bitIdx < bits.length; y++) {
    for (let x = wEven / 2; x < wEven && bitIdx < bits.length; x++) {
      coeff[y][x] += bits[bitIdx] === 1 ? delta : -delta;
      bitIdx++;
    }
  }
  const recovered = haarIdwt(coeff);
  for (let y = 0; y < hEven; y++) {
    for (let x = 0; x < wEven; x++) {
      luma[y][x] = recovered[y][x];
    }
  }
  log(`DWT 水印写入 ${bitIdx} bits`);
  return applyLumaToImageData(imageData, luma, baseLuma);
}

function extractDWT(imageData, bitCount = 0) {
  const luma = toLuma(imageData);
  const hEven = luma.length - (luma.length % 2);
  const wEven = luma[0].length - (luma[0].length % 2);
  const trimmed = luma.slice(0, hEven).map(row => row.slice(0, wEven));
  const coeff = haarDwt(trimmed);
  const bits = [];
  for (let y = hEven / 2; y < hEven; y++) {
    for (let x = wEven / 2; x < wEven; x++) {
      bits.push(coeff[y][x] > 0 ? 1 : 0);
      if (bitCount && bits.length >= bitCount) return bits;
    }
  }
  return bits;
}

function positionsPerBit(pixels, bitCount) {
  return Math.max(12, Math.floor(pixels / Math.max(bitCount, 1) / 6));
}

function embedSpatial(imageData, bits, strength = 6, seed = 'rbwm') {
  const { data, width, height } = imageData;
  const pixels = width * height;
  const rand = seededRandom(seed + '-spatial');
  const amplitude = 0.8 * strength;
  const countPerBit = positionsPerBit(pixels, bits.length);
  for (let bitIndex = 0; bitIndex < bits.length; bitIndex++) {
    const bit = bits[bitIndex] ? 1 : -1;
    for (let i = 0; i < countPerBit; i++) {
      const pos = Math.floor(rand() * pixels);
      const idx = pos * 4;
      const pn = rand() > 0.5 ? 1 : -1;
      const delta = amplitude * pn * bit;
      data[idx] = clamp(data[idx] + delta);
      data[idx + 1] = clamp(data[idx + 1] + delta);
      data[idx + 2] = clamp(data[idx + 2] + delta);
    }
  }
  log(`空域扩频写入 ${bits.length} bits，扩展度 ${countPerBit}`);
  return imageData;
}

function extractSpatialBits(imageData, bitCount, seed) {
  const { data, width, height } = imageData;
  const pixels = width * height;
  const rand = seededRandom(seed + '-spatial');
  const countPerBit = positionsPerBit(pixels, bitCount);
  const bits = [];
  for (let bitIndex = 0; bitIndex < bitCount; bitIndex++) {
    let accum = 0;
    for (let i = 0; i < countPerBit; i++) {
      const pos = Math.floor(rand() * pixels);
      const idx = pos * 4;
      const pn = rand() > 0.5 ? 1 : -1;
      const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      accum += (lum - 128) * pn;
    }
    bits.push(accum > 0 ? 1 : 0);
  }
  return bits;
}

function extractSpatial(imageData, bitCount, seed = 'rbwm') {
  if (bitCount && bitCount > 0) {
    return extractSpatialBits(imageData, bitCount, seed);
  }
  // two-pass: first decode长度头，再按总长度重新采样，保证写入/读取扩展度一致
  const header = extractSpatialBits(imageData, 16, seed);
  let length = 0;
  header.forEach(b => { length = (length << 1) | b; });
  const totalBits = 16 + length * 8;
  return extractSpatialBits(imageData, totalBits, seed);
}

function drawImageData(canvas, imageData) {
  if (!imageData) return;
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
}

async function handleCoverInput(file) {
  if (!file) return;
  coverImageData = await loadImageToCanvas(file, coverPreview);
  log(`载入原图：${file.name} (${coverImageData.width}×${coverImageData.height})`);
}

async function handleStegoInput(file) {
  if (!file) return;
  stegoImageData = await loadImageToCanvas(file, stegoPreview);
  log(`载入待检测图：${file.name}`);
}

function ensureBitsCapacity(bits, imageData) {
  const blockCount = Math.floor(imageData.width / 8) * Math.floor(imageData.height / 8);
  const dwtCount = Math.floor(imageData.width / 2) * Math.floor(imageData.height / 2);
  const spatialCount = bits.length; // spread spectrum repeats internally
  return bits.length <= blockCount + dwtCount + spatialCount;
}

async function embedWatermark() {
  if (!coverImageData) {
    alert('请先选择原始图片');
    return;
  }
  const message = messageInput.value.trim();
  if (!message) {
    alert('请输入要隐藏的文字');
    return;
  }
  const bits = textToBits(message);
  if (!ensureBitsCapacity(bits, coverImageData)) {
    alert('图片过小或文字过长，请减少字数');
    return;
  }
  const strength = Number(strengthInput.value) || 6;
  const seed = seedInput.value || 'rbwm';
  let out = imageDataCopy(coverImageData);
  if (document.getElementById('algo-dct').checked) out = embedDCT(out, bits, strength);
  if (document.getElementById('algo-dwt').checked) out = embedDWT(out, bits, strength);
  if (document.getElementById('algo-spatial').checked) out = embedSpatial(out, bits, strength, seed);
  stegoImageData = out;
  drawImageData(stegoPreview, out);
  downloadBtn.disabled = false;
  log('水印写入完成，可下载或直接检测');
}

function downloadStego() {
  if (!stegoImageData) return;
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = stegoImageData.width;
  tmpCanvas.height = stegoImageData.height;
  const ctx = tmpCanvas.getContext('2d');
  ctx.putImageData(stegoImageData, 0, 0);
  const url = tmpCanvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url;
  a.download = 'watermarked.png';
  a.click();
}

function autoBitCount(bits) {
  if (bits.length < 16) return 0;
  let len = 0;
  for (let i = 0; i < 16; i++) len = (len << 1) | bits[i];
  return 16 + len * 8;
}

function extractFromImage(imageData) {
  const seed = detectSeedInput.value || 'rbwm';
  const forcedLen = Number(detectLenInput.value || 0);
  const selected = {
    dct: document.getElementById('detect-dct').checked,
    dwt: document.getElementById('detect-dwt').checked,
    spatial: document.getElementById('detect-spatial').checked,
  };
  const results = [];
  const hiddenOutputs = [];
  if (selected.dct) {
    const raw = extractDCT(imageData, forcedLen ? forcedLen * 8 + 16 : 0);
    const lenBits = forcedLen ? forcedLen * 8 + 16 : autoBitCount(raw);
    const msg = bitsToText(raw.slice(0, lenBits), forcedLen);
    const gibberish = isLikelyGibberish(msg);
    if (gibberish) {
      hiddenOutputs.push(`DCT 提取（疑似乱码）:\n${msg}`);
      results.push('DCT 提取：疑似乱码，已隐藏至下方调试框');
    } else {
      results.push(`DCT 提取：${msg || '[未解析到内容]'}`);
    }
  }
  if (selected.dwt) {
    const raw = extractDWT(imageData, forcedLen ? forcedLen * 8 + 16 : 0);
    const lenBits = forcedLen ? forcedLen * 8 + 16 : autoBitCount(raw);
    const msg = bitsToText(raw.slice(0, lenBits), forcedLen);
    const gibberish = isLikelyGibberish(msg);
    if (gibberish) {
      hiddenOutputs.push(`DWT 提取（疑似乱码）:\n${msg}`);
      results.push('DWT 提取：疑似乱码，已隐藏至下方调试框');
    } else {
      results.push(`DWT 提取：${msg || '[未解析到内容]'}`);
    }
  }
  if (selected.spatial) {
    const raw = extractSpatial(imageData, forcedLen ? forcedLen * 8 + 16 : 0, seed);
    const lenBits = forcedLen ? forcedLen * 8 + 16 : autoBitCount(raw);
    const msg = bitsToText(raw.slice(0, lenBits), forcedLen);
    const gibberish = isLikelyGibberish(msg);
    if (gibberish) {
      hiddenOutputs.push(`空域扩频提取（疑似乱码）:\n${msg}`);
      results.push('空域扩频提取：疑似乱码，已隐藏至下方调试框');
    } else {
      results.push(`空域扩频提取：${msg || '[未解析到内容]'}`);
    }
  }
  detectResult.value = results.join('\n');
  if (debugOutput) {
    debugOutput.value = hiddenOutputs.length
      ? hiddenOutputs.join('\n\n---\n\n')
      : '暂无隐藏输出或乱码内容';
  }
  log('检测完成');
}

function currentDetectImage() {
  if (stegoImageData) return stegoImageData;
  if (coverImageData) return coverImageData;
  return null;
}

coverInput.addEventListener('change', async e => {
  if (e.target.files.length) await handleCoverInput(e.target.files[0]);
});
stegoInput.addEventListener('change', async e => {
  if (e.target.files.length) await handleStegoInput(e.target.files[0]);
});
embedBtn.addEventListener('click', embedWatermark);
downloadBtn.addEventListener('click', downloadStego);
detectBtn.addEventListener('click', () => {
  const img = currentDetectImage();
  if (!img) { alert('请先上传或生成含水印图片'); return; }
  extractFromImage(img);
});

log('准备就绪：上传图片后即可写入/检测水印');
