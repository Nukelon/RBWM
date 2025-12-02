const originalCanvas = document.getElementById("originalCanvas");
const resultCanvas = document.getElementById("resultCanvas");
const originalCtx = originalCanvas.getContext("2d");
const resultCtx = resultCanvas.getContext("2d");
const logArea = document.getElementById("log");

const controls = {
  imageInput: document.getElementById("imageInput"),
  message: document.getElementById("message"),
  seed: document.getElementById("seed"),
  strength: document.getElementById("strength"),
  repeat: document.getElementById("repeat"),
  useDct: document.getElementById("useDct"),
  useDwt: document.getElementById("useDwt"),
  useSpatial: document.getElementById("useSpatial"),
  embed: document.getElementById("embed"),
  decode: document.getElementById("decode"),
  download: document.getElementById("download"),
};

function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  logArea.textContent += `[${timestamp}] ${message}\n`;
  logArea.scrollTop = logArea.scrollHeight;
}

function clamp(v) {
  return Math.max(0, Math.min(255, v));
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function encodeTextToBits(text) {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  const length = bytes.length;
  const bits = [];
  for (let i = 0; i < 16; i++) {
    bits.push((length >> (15 - i)) & 1);
  }
  for (const byte of bytes) {
    for (let i = 7; i >= 0; i--) {
      bits.push((byte >> i) & 1);
    }
  }
  return bits;
}

function bitsToText(bits) {
  if (bits.length < 16) return "";
  let length = 0;
  for (let i = 0; i < 16; i++) {
    length = (length << 1) | bits[i];
  }
  const expectedBits = 16 + length * 8;
  if (bits.length < expectedBits) return "";
  const bytes = [];
  for (let i = 16; i < expectedBits; i += 8) {
    let byte = 0;
    for (let j = 0; j < 8; j++) {
      byte = (byte << 1) | bits[i + j];
    }
    bytes.push(byte);
  }
  try {
    return new TextDecoder().decode(new Uint8Array(bytes));
  } catch (err) {
    return "";
  }
}

function loadToCanvas(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const ratio = Math.min(originalCanvas.width / img.width, originalCanvas.height / img.height, 1);
        const w = Math.floor(img.width * ratio);
        const h = Math.floor(img.height * ratio);
        [originalCanvas, resultCanvas].forEach((canvas) => {
          canvas.width = w;
          canvas.height = h;
        });
        originalCtx.drawImage(img, 0, 0, w, h);
        resultCtx.drawImage(img, 0, 0, w, h);
        resolve();
      };
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function copyOriginalToResult() {
  const imgData = originalCtx.getImageData(0, 0, originalCanvas.width, originalCanvas.height);
  resultCtx.putImageData(imgData, 0, 0);
}

function embedSpatial(data, bits, options) {
  const { seed, repeat } = options;
  const rng = mulberry32(seed);
  const totalPixels = data.data.length / 4;
  const repetitions = Math.max(1, repeat);
  for (let r = 0; r < repetitions; r++) {
    bits.forEach((bit) => {
      const pos = Math.floor(rng() * totalPixels);
      const idx = pos * 4 + 2; // blue channel
      const current = data.data[idx];
      const targetParity = bit & 1;
      if ((current & 1) !== targetParity) {
        data.data[idx] = clamp(current + (targetParity ? 1 : -1));
      }
      // push contrast slightly to resist JPEG rounding
      data.data[idx] = clamp(data.data[idx] + (targetParity ? 2 : -2));
    });
  }
  return data;
}

function decodeSpatial(data, bitsLength, options) {
  const { seed, repeat } = options;
  const rng = mulberry32(seed);
  const totalPixels = data.data.length / 4;
  const repetitions = Math.max(1, repeat);
  const votes = Array(bitsLength).fill(0);
  for (let r = 0; r < repetitions; r++) {
    for (let i = 0; i < bitsLength; i++) {
      const pos = Math.floor(rng() * totalPixels);
      const idx = pos * 4 + 2;
      const bit = data.data[idx] & 1;
      votes[i] += bit ? 1 : -1;
    }
  }
  return votes.map((v) => (v >= 0 ? 1 : 0));
}

function dct2(block) {
  const N = 8;
  const result = Array.from({ length: N }, () => Array(N).fill(0));
  const c = (x) => (x === 0 ? 1 / Math.sqrt(2) : 1);
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let sum = 0;
      for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
          sum += block[x][y] * Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N)) * Math.cos(((2 * y + 1) * v * Math.PI) / (2 * N));
        }
      }
      result[u][v] = 0.25 * c(u) * c(v) * sum;
    }
  }
  return result;
}

function idct2(coeff) {
  const N = 8;
  const result = Array.from({ length: N }, () => Array(N).fill(0));
  const c = (x) => (x === 0 ? 1 / Math.sqrt(2) : 1);
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      let sum = 0;
      for (let u = 0; u < N; u++) {
        for (let v = 0; v < N; v++) {
          sum += c(u) * c(v) * coeff[u][v] * Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N)) * Math.cos(((2 * y + 1) * v * Math.PI) / (2 * N));
        }
      }
      result[x][y] = 0.25 * sum;
    }
  }
  return result;
}

function embedDct(data, bits, options) {
  const { strength } = options;
  const w = data.width;
  const h = data.height;
  const view = data.data;
  const blockCount = Math.floor(w / 8) * Math.floor(h / 8);
  if (blockCount < bits.length) {
    log("DCT 容量不足，部分信息会被截断。");
  }
  let bitIndex = 0;
  for (let by = 0; by <= h - 8 && bitIndex < bits.length; by += 8) {
    for (let bx = 0; bx <= w - 8 && bitIndex < bits.length; bx += 8) {
      const block = Array.from({ length: 8 }, (_, x) =>
        Array.from({ length: 8 }, (_, y) => view[((by + x) * w + (bx + y)) * 4 + 2] - 128)
      );
      const coeff = dct2(block);
      const bit = bits[bitIndex++];
      const pos = [3, 4];
      const base = coeff[pos[0]][pos[1]];
      const q = Math.max(4, strength);
      const anchor = Math.round(base / q) * q;
      coeff[pos[0]][pos[1]] = anchor + (bit ? q * 0.7 : q * 0.2);
      const restored = idct2(coeff);
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          const idx = ((by + x) * w + (bx + y)) * 4 + 2;
          view[idx] = clamp(restored[x][y] + 128);
        }
      }
    }
  }
  return data;
}

function decodeDct(data, bitsLength, options) {
  const { strength } = options;
  const w = data.width;
  const h = data.height;
  const view = data.data;
  const bits = [];
  let collected = 0;
  for (let by = 0; by <= h - 8 && collected < bitsLength; by += 8) {
    for (let bx = 0; bx <= w - 8 && collected < bitsLength; bx += 8) {
      const block = Array.from({ length: 8 }, (_, x) =>
        Array.from({ length: 8 }, (_, y) => view[((by + x) * w + (bx + y)) * 4 + 2] - 128)
      );
      const coeff = dct2(block);
      const q = Math.max(4, strength);
      const pos = [3, 4];
      const mod = coeff[pos[0]][pos[1]] % q;
      bits.push(mod > q * 0.45 ? 1 : 0);
      collected++;
    }
  }
  return bits;
}

function haarDwt(matrix, width, height) {
  const temp = [];
  // horizontal
  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x += 2) {
      const a = matrix[y][x];
      const b = matrix[y][x + 1];
      row.push((a + b) / 2);
    }
    for (let x = 0; x < width; x += 2) {
      const a = matrix[y][x];
      const b = matrix[y][x + 1];
      row.push((a - b) / 2);
    }
    temp.push(row);
  }
  // vertical
  const result = Array.from({ length: height }, () => Array(width).fill(0));
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y += 2) {
      const a = temp[y][x];
      const b = temp[y + 1][x];
      result[y / 2][x] = (a + b) / 2;
      result[height / 2 + y / 2][x] = (a - b) / 2;
    }
  }
  return result;
}

function inverseHaar(coeff, width, height) {
  const temp = Array.from({ length: height }, () => Array(width).fill(0));
  // vertical inverse
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height / 2; y++) {
      const a = coeff[y][x];
      const b = coeff[height / 2 + y][x];
      temp[2 * y][x] = a + b;
      temp[2 * y + 1][x] = a - b;
    }
  }
  const result = Array.from({ length: height }, () => Array(width).fill(0));
  // horizontal inverse
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width / 2; x++) {
      const a = temp[y][x];
      const b = temp[y][width / 2 + x];
      result[y][2 * x] = a + b;
      result[y][2 * x + 1] = a - b;
    }
  }
  return result;
}

function embedDwt(data, bits, options) {
  const { strength } = options;
  const w = data.width - (data.width % 2);
  const h = data.height - (data.height % 2);
  const matrix = Array.from({ length: h }, (_, y) =>
    Array.from({ length: w }, (_, x) => data.data[(y * data.width + x) * 4 + 2] - 128)
  );
  const coeff = haarDwt(matrix, w, h);
  const capacity = (w * h) / 4;
  if (capacity < bits.length) {
    log("DWT 容量不足，部分信息会被截断。");
  }
  let bitIndex = 0;
  const startRow = h / 2;
  for (let y = startRow; y < h && bitIndex < bits.length; y++) {
    for (let x = 0; x < w && bitIndex < bits.length; x++) {
      const bit = bits[bitIndex++];
      const q = Math.max(4, strength);
      const anchor = Math.round(coeff[y][x] / q) * q;
      coeff[y][x] = anchor + (bit ? q * 0.65 : q * 0.2);
    }
  }
  const restored = inverseHaar(coeff, w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * data.width + x) * 4 + 2;
      data.data[idx] = clamp(restored[y][x] + 128);
    }
  }
  return data;
}

function decodeDwt(data, bitsLength, options) {
  const { strength } = options;
  const w = data.width - (data.width % 2);
  const h = data.height - (data.height % 2);
  const matrix = Array.from({ length: h }, (_, y) =>
    Array.from({ length: w }, (_, x) => data.data[(y * data.width + x) * 4 + 2] - 128)
  );
  const coeff = haarDwt(matrix, w, h);
  const bits = [];
  const q = Math.max(4, strength);
  const startRow = h / 2;
  for (let y = startRow; y < h && bits.length < bitsLength; y++) {
    for (let x = 0; x < w && bits.length < bitsLength; x++) {
      const mod = coeff[y][x] % q;
      bits.push(mod > q * 0.45 ? 1 : 0);
    }
  }
  return bits;
}

function ensureImageLoaded() {
  const data = originalCtx.getImageData(0, 0, originalCanvas.width, originalCanvas.height);
  if (data.data.every((v) => v === 0)) {
    log("请先加载图片。");
    return false;
  }
  return true;
}

function embed() {
  if (!ensureImageLoaded()) return;
  copyOriginalToResult();
  const message = controls.message.value || "RBWM watermark";
  const bits = encodeTextToBits(message);
  const options = {
    seed: Number(controls.seed.value) || 1,
    strength: Number(controls.strength.value) || 12,
    repeat: Number(controls.repeat.value) || 4,
  };
  let data = resultCtx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
  const enabled = [];
  if (controls.useSpatial.checked) {
    data = embedSpatial(data, bits, options);
    enabled.push("空域伪随机");
  }
  if (controls.useDct.checked) {
    data = embedDct(data, bits, options);
    enabled.push("DCT 频域");
  }
  if (controls.useDwt.checked) {
    data = embedDwt(data, bits, options);
    enabled.push("DWT 频域");
  }
  resultCtx.putImageData(data, 0, 0);
  log(`完成嵌入，算法：${enabled.join(" / ")}；信息位数：${bits.length}`);
}

function decode() {
  if (!ensureImageLoaded()) return;
  const options = {
    seed: Number(controls.seed.value) || 1,
    strength: Number(controls.strength.value) || 12,
    repeat: Number(controls.repeat.value) || 4,
  };
  const data = resultCtx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
  const targetLength = 16 + Math.max(1, (controls.message.value || "测试").length) * 8;
  const results = [];
  if (controls.useSpatial.checked) {
    const bits = decodeSpatial(data, targetLength, options);
    results.push({ name: "空域伪随机", text: bitsToText(bits) });
  }
  if (controls.useDct.checked) {
    const bits = decodeDct(data, targetLength, options);
    results.push({ name: "DCT 频域", text: bitsToText(bits) });
  }
  if (controls.useDwt.checked) {
    const bits = decodeDwt(data, targetLength, options);
    results.push({ name: "DWT 频域", text: bitsToText(bits) });
  }
  const fusion = results
    .map((r) => r.text)
    .filter((t) => t)
    .reduce((acc, cur, _, arr) => {
      const count = arr.filter((t) => t === cur).length;
      return count > (acc.count || 0) ? { text: cur, count } : acc;
    }, {}).text || "";
  const lines = results
    .map((r) => `${r.name}: ${r.text || "<解码失败>"}`)
    .join("\n");
  log(`提取结果：\n${lines}\n融合结论：${fusion || "未能可靠解码"}`);
}

function download() {
  if (!ensureImageLoaded()) return;
  const link = document.createElement("a");
  link.href = resultCanvas.toDataURL("image/png");
  link.download = "watermarked.png";
  link.click();
}

controls.imageInput.addEventListener("change", (e) => {
  const [file] = e.target.files;
  if (!file) return;
  loadToCanvas(file)
    .then(() => log(`已加载图片：${file.name} (${originalCanvas.width}x${originalCanvas.height})`))
    .catch(() => log("图片加载失败"));
});

controls.embed.addEventListener("click", embed);
controls.decode.addEventListener("click", decode);
controls.download.addEventListener("click", download);

log("准备就绪：请加载图片并选择要使用的水印算法。");
