import { Matrix, zeros, subtract, matrix, subset, index, norm } from "mathjs";
import { mixtureGaussianParameter, logGmmKernel } from "../gmm/mathExtension";
import { fittingGMM_T } from "../gmm/gmm";
import { mincut } from "./mincut";


const GmmKernelNum = 5;

type rgb = readonly [number, number, number];
type rgba = readonly [number, number, number, number];

type img<T extends rgb|rgba> = {
  readonly data: ReadonlyArray<T>,
  readonly width: number,
  readonly height: number,
  readonly channel: T["length"]
}

type foreground = 1;
type background = 0;

const foregroundFlag = 1;
const backgroundFlag = 0;

type imgWithMask<T extends rgb|rgba> = {
  readonly mask: ReadonlyArray<foreground|background>
} & img<T>;


export function grabcut(
  input_img: ImageData,
  foreground_indices: ReadonlyArray<number>,
  background_indices: ReadonlyArray<number>
): imgWithMask<rgba> {
  const seg_input_img = convert1dRGBAImageArrayToRGBImg(input_img.data, input_img.width, input_img.height);

  const maybe_foreground = seg_input_img.data.filter((_, index) => !background_indices.includes(index));
  const background = background_indices.map(i => seg_input_img.data[i]);

  const mask = execSegmentation(
    seg_input_img,
    maybe_foreground,
    background,
    foreground_indices,
    background_indices
  );

  return grabcutRec(
    seg_input_img,
    foreground_indices,
    background_indices,
    mask
  );
}


function grabcutRec(
  input_img: img<rgb>,
  foreground_constraints: ReadonlyArray<number>,
  background_constraints: ReadonlyArray<number>,
  preResultMask: (foreground|background)[]
): imgWithMask<rgba> {
  const {foregrounds, backgrounds} = preResultMask.reduce((pre, current, i) => {
    if (current === backgroundFlag) {
      pre.backgrounds.push(input_img.data[i]);
    }else {
      pre.foregrounds.push(input_img.data[i]);
    }

    return pre;
  }, {
    foregrounds: [] as rgb[],
    backgrounds: [] as rgb[]
  });

  const mask = execSegmentation(
    input_img,
    foregrounds,
    backgrounds,
    foreground_constraints,
    background_constraints
  );

  if (isDifferentMask(mask, preResultMask)) {
    return grabcutRec(
      input_img,
      foreground_constraints,
      background_constraints,
      mask
    );
  }

  return {
    ...convertRGBImgToRGBAImg(input_img),
    mask: mask
  }
}


function execSegmentation(
  input_img: img<rgb>,
  foregrounds: readonly rgb[],
  backgrounds: readonly rgb[],
  foreground_constraints: readonly number[],
  background_constraints: readonly number[]
): (foreground|background)[] {
  const foreground_model = fittingGMM_T(foregrounds, GmmKernelNum);
  const background_model = fittingGMM_T(backgrounds, GmmKernelNum);

  const {adjacencyMatrix, sourceIndex, terminalIndex} = makeAdjacencyMatrixForFlow(
    input_img,
    foreground_model,
    background_model,
    foreground_constraints,
    background_constraints
  );
  
  const mincutResult = mincut(adjacencyMatrix, sourceIndex, terminalIndex);

  const mask = Array<foreground|background>(input_img.data.length).fill(backgroundFlag);
  mincutResult.source.forEach(pixelIndex => mask[pixelIndex] = foregroundFlag);

  return mask;
}


function isDifferentMask(
  mask1: (foreground|background)[], mask2: (foreground|background)[]  
): boolean {
  if (mask1.length !== mask2.length) {
    throw new Error(`maskの大きさが異なります  1: ${mask1.length}, 2: ${mask2.length}`);
  }

  for (let i = 0; i < mask1.length; i++) {
    if (mask1[i] !== mask2[i]) {
      return true;
    }
  }

  return false;
}


//1次元のRGBA配列の画像データをRGBベクターの配列形式に変換する。
//canvasにレンダリングされている画像を取得するとRGBAの1次元配列になっているため変換が必要。
function convert1dRGBAImageArrayToRGBImg(arrayRGBA: Uint8ClampedArray, width: number, height: number): img<rgb> {
  const arr = Array(arrayRGBA.length / 4).fill(null).map<rgb>( (_: any, i: number) => {
    const r = arrayRGBA[4 * i];
    const g = arrayRGBA[4 * i + 1];
    const b = arrayRGBA[4 * i + 2];
  
    return [r, g, b];
  });

  return {
    data: arr,
    channel: 3,
    width,
    height
  }
}


function convertRGBImgToRGBAImg(source: img<rgb>): img<rgba> {
  return {
    data: source.data.map(rgb => [...rgb, 1]),
    width: source.width,
    height: source.height,
    channel: 4
  }
}


//推定されたGMMと入力画像からフローの隣接行列を作成
function makeAdjacencyMatrixForFlow(
  input_img: img<rgb>,
  foregroundGmm: {
    p: mixtureGaussianParameter, 
    cluster: ReadonlyArray<number> 
  },
  backgroundGmm: {
    p: mixtureGaussianParameter, 
    cluster: ReadonlyArray<number> 
  },
  confirmedForegrounds: readonly number[],
  confirmedBackgrounds: readonly number[],
): {
  adjacencyMatrix: Matrix,
  sourceIndex: 0,
  terminalIndex: 1
} {
  const adjacencyMatrix = zeros([input_img.height+2, input_img.width+2], "sparse") as Matrix;
  //row, col = 0がソース、row, col = 1がターミナル
  const foregroundDataTerm = matrix(
    calcDataTerm(input_img.data, foregroundGmm, confirmedForegrounds)
  );

  const backgroundDataTerm = matrix(
    calcDataTerm(input_img.data, backgroundGmm, confirmedBackgrounds)
  );

  const vTerm = calcV(input_img);
  const indices = input_img.data.map((_, i) => i+2);
  subset(
    adjacencyMatrix,
    index(indices, indices),
    vTerm
  );
  
  subset(
    adjacencyMatrix,
    index(0, indices),
    foregroundDataTerm
  );

  subset(
    adjacencyMatrix,
    index(indices, 1),
    backgroundDataTerm
  );

  return {
    adjacencyMatrix,
    sourceIndex: 0,
    terminalIndex: 1
  };
}


function calcDataTerm(
  RGBArray: ReadonlyArray<rgb>, 
  gmm: {
    p: mixtureGaussianParameter, 
    cluster: ReadonlyArray<number> 
  },
  //コストが無限大になる画素をインデックス指定
  confirmeds: readonly number[]
): number[] {
  //TODO: includesとmapを直列にして計算量のオーダーを減らしたい
  return RGBArray.map((rgbVector, index) => {
    if (confirmeds.includes(index)) {
      return Number.MAX_SAFE_INTEGER;
    }
    
    const kernelI = gmm.cluster[index];
    const dataTerm = logGmmKernel(
      matrix(rgbVector as [number, number, number]),
      gmm.p.u[kernelI],
      gmm.p.sigma[kernelI], 
      gmm.p.pi[kernelI], 
    );
    return Math.floor(10000*dataTerm); //整数でないとアルゴリズムが停止しない可能性がある    
  }, []);
}

function isTop(imgWdith: number, i: number) {
  return i - imgWdith < 0
}

function isBottom(imgWdith: number, imgLength: number, i: number) {
  return i + imgWdith >= imgLength
}

function isLeft(imgWdith: number, i: number) {
  return i % imgWdith === 0
}

function isRight(imgWdith: number, i: number) {
  return i % imgWdith === imgWdith - 1
}


//入力のmatに副作用を及ぼすため注意
function setEValue(array: ReadonlyArray<rgb>, from: number, to: number, mat: Matrix) {
  const value = norm(
    subtract(
      matrix(array[from] as [number, number, number]),
      matrix(array[to] as [number, number, number])
    ) as Matrix
  ) as number;

  subset(
    mat,
    index(from, to),
    Math.floor(1000*value) //整数でないとアルゴリズムが停止しない可能性がある
  );
}


function calcV(
  input: img<rgb>
): Matrix {
  const mat = zeros(input.data.length, input.data.length, "sparse") as Matrix;
  
  input.data.forEach((rgbVector, scopeIndex) => {
    const set = (dstIndex: number) => setEValue(input.data, scopeIndex, dstIndex, mat);
    //上端でないなら
    if (!isTop(input.width, scopeIndex)) {
      set(scopeIndex - input.width);
    }

    //下端でない
    if (!isBottom(input.width, input.data.length, scopeIndex)) {
      set(scopeIndex + input.width);
    }

    //右端でない
    if (!isRight(input.width, scopeIndex)) {
      set(scopeIndex + 1);
    }

    //左端でない
    if (!isLeft(input.width, scopeIndex)) {
      set(scopeIndex - 1);
    }

    //左上でない
    if (!(isTop(input.width, scopeIndex) || isLeft(input.width, scopeIndex))) {
      set(scopeIndex - 1 - input.width);
    }

    //右上でない
    if (!(isTop(input.width, scopeIndex) || isRight(input.width, scopeIndex))) {
      set(scopeIndex + 1 - input.width);
    }

    if (!(isBottom(input.width, input.data.length, scopeIndex) || isLeft(input.width, scopeIndex))) {
      set(scopeIndex - 1 + input.width);
    }

    if (!(isBottom(input.width, input.data.length, scopeIndex) || isRight(input.width, scopeIndex))) {
      set(scopeIndex + 1 + input.width);
    }
  });

  return mat;
}