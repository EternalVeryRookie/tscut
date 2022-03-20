import {
  Matrix, subtract, multiply, divide, transpose, index, matrix, add, log, zeros, randomInt, abs
} from "mathjs";

import { covMat, means, mixtureGaussianParameter, gaussianXd, mixtureGaussian, getRow } from "./mathExtension";
import { generateSerialNumbers } from "../libs/array";



//データがどのクラスタに所属するかを推定
//各ガウスカーネルに対して入力xにおける値を計算して、最も値が大きいカーネルにデータが所属しているとする
function detectCluster(x: Matrix, p: mixtureGaussianParameter): number {
  const init = p.pi[0]*gaussianXd(x, {u: p.u[0], sigma: p.sigma[0]});
  return generateSerialNumbers(p.u.length).reduce<{value: number, index: number}>((max, index) => {
    if (index === 0) {
      return max;
    }

    const prob = p.pi[index] * gaussianXd(x, {u: p.u[index], sigma: p.sigma[index]})
    if (prob > max.value) {
      return {
        value: prob,
        index: index
      }
    }

    return max;
  }, {value: init, index: 0}).index
}

type FixLengthArray<N extends number, T> = FixLengthArrayRec<N, T, []>;

type FixLengthArrayRec<Num, Elm, T extends readonly unknown[]> =
  T extends { length: Num }
    ? T
    : FixLengthArrayRec<Num, Elm, readonly [...T, Elm]>;


export const fittingGMM_T = <T extends number>(data: ReadonlyArray<FixLengthArray<T, number>>, k: number) => fittingGMM(data, k);

export function fittingGMM(data: ReadonlyArray<ReadonlyArray<number>>, k: number): 
  {
    //推定された混合ガウス分布のパラメーター。
    p: mixtureGaussianParameter, 

    //入力データがどのクラスタに所属するかを表す配列。cluster[i] == k のとき、i番目のデータがk番目のガウスカーネルに所属しているとする。
    cluster: ReadonlyArray<number> 
  } {
  const initClusters = data.map(() => randomInt(0, k));
  const initParameters: mixtureGaussianParameter = generateSerialNumbers(k).reduce<{u: Matrix[], sigma: Matrix[], pi: number[]}>((gmmParam, clusterIndex) => {
    const clusterData = initClusters.flatMap((s, i) => s === clusterIndex ? i : []);
    const clusterDataMatrix = matrix(clusterData.map(i => data[i]) as number[][]);
    const mean = means(clusterDataMatrix);
    const cov = covMat(clusterDataMatrix);
    const pi = clusterDataMatrix.size()[0]/ data.length;

    gmmParam.u.push(mean);
    gmmParam.sigma.push(cov);
    gmmParam.pi.push(pi);
    return gmmParam;
  }, {
    u: [], sigma: [], pi: []
  });
  
  const dataMat = matrix(data as number[][]);
  const initLoglike = calcLogLikelihood(dataMat, initParameters);
  const threshold = 0.1;
  const gmmParam = fittingGMMRec(dataMat, initParameters, initLoglike, threshold);

  return {
    p: gmmParam,
    cluster: Array(dataMat.size()[0]).fill(null).map((_, row) => detectCluster(getRow(dataMat, row), gmmParam))
  }
}


function fittingGMMRec(data: Matrix, gmmParam: mixtureGaussianParameter, loglikehood: number, epsilon: number): mixtureGaussianParameter {
  //contirubute[n][k] = γ_nk
  const contirubutes = generateSerialNumbers(data.size()[0]).map(dataI => {
    return generateSerialNumbers(gmmParam.u.length).map(clusterI => {
      return calcContributionRatio(getRow(data, dataI), clusterI, gmmParam);
    })
  });

  const newGmmParam = generateSerialNumbers(gmmParam.pi.length).reduce<{u: Matrix[], sigma: Matrix[], pi: number[]}>((p, clusterK) => {
    const N_k = contirubutes.reduce<number>((total, current) => current[clusterK] + total, 0);

    const newMean = divide(
      contirubutes.reduce<Matrix>((total, contoribute_n, n) => {
        return add(total, multiply(contoribute_n[clusterK], getRow(data, n))) as Matrix
      }, zeros([1, data.size()[1]]) as Matrix)
    , N_k
    ) as Matrix;

    const newSigma = divide(
      contirubutes.reduce<Matrix>((total, contoribute_n, n) => {
        const delta = subtract(getRow(data, n), gmmParam.u[clusterK]) as Matrix;
        const tmp = multiply(contoribute_n[clusterK], multiply(transpose(delta), delta));
        return add(total, tmp) as Matrix;
      }, zeros([data.size()[1], data.size()[1]]) as Matrix)
    , N_k
    ) as Matrix;

    const pi = N_k / data.size()[0];

    p.u.push(newMean);
    p.sigma.push(newSigma);
    p.pi.push(pi);

    return p;
  },  {
    u: [], sigma: [], pi: []
  });

  const newLogLikelihood = calcLogLikelihood(data, newGmmParam);
  if (abs(newLogLikelihood - loglikehood) < epsilon) {
    return newGmmParam;
  }
  
  return fittingGMMRec(data, newGmmParam, newLogLikelihood, epsilon);
}


//負担率
function calcContributionRatio(x: Matrix, k: number, p: mixtureGaussianParameter): number {
  const a = mixtureGaussian(x, p);
  const b = p.pi[k] * gaussianXd(x, {u: p.u[k], sigma: p.sigma[k]});
  return b / a;
}


//dataは各実現値を行に設定した行列（つまり、データの数が行数で、データの次元が列数になる）
function calcLogLikelihood(data: Matrix, p: mixtureGaussianParameter): number {
  return Array(data.size()[0]).fill(null).reduce((pre, _, i) => {
    const vec = data.subset(index(i, [...Array(data.size()[1])].map((_, i) => i)));

    return pre + log(mixtureGaussian(vec, p));
  }, 0);
}
