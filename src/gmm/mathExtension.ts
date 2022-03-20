//mathの拡張モジュール。多次元ガウス関数の計算や共分散行列の計算など

import { Matrix, add, divide, dot, matrix, dotDivide, map, subtract, index, zeros, inv, exp, multiply, transpose, sqrt, pow, pi, det, log } from "mathjs";
import { generateSerialNumbers } from "../libs/array";


export type gaussianParameter = {
  readonly u: Matrix,
  readonly sigma: Matrix
}


export type mixtureGaussianParameter = {
  readonly u: ReadonlyArray<Matrix>,
  readonly sigma: ReadonlyArray<Matrix>,
  readonly pi: ReadonlyArray<number>
}


export function gaussianXd(x: Matrix, p: gaussianParameter): number {
  const tmp = subtract(x, p.u) as Matrix;
  const invMat = inv(p.sigma);
  const expValue = exp(multiply(divide(dot(transpose(multiply(tmp, invMat)), transpose(tmp)), 2), -1) as number);
  const coef = sqrt(multiply(pow(2 * pi, p.u.size()[0]), det(p.sigma)) as number)

  return divide(expValue, coef) as number;
}


export function mixtureGaussian(x: Matrix, p: mixtureGaussianParameter): number {
  return Array(p.u.length).fill(null).reduce((pre, _, index) => {
    return pre + p.pi[index] * gaussianXd(x, {u: p.u[index], sigma: p.sigma[index]})
  }, 0);
}


export function logGmmKernel(x: Matrix, u: Matrix, sigma: Matrix, pi: number): number {
  const tmp = subtract(x, u) as Matrix;
  const invMat = inv(sigma);
  return -log(pi) + log(det(sigma))/2 + divide(dot(transpose(multiply(tmp, invMat)), transpose(tmp)), 2)
}


export function covMat(data: Matrix): Matrix {
  const mean = means(data);
  const deviationMatrix = deviationMat(data, mean);
  const array2d = generateSerialNumbers(data.size()[1]).map(covMatRow => {
    return generateSerialNumbers(data.size()[1]).map(covMatCol => {
      const a = getCol(deviationMatrix, covMatRow);
      const b = getCol(deviationMatrix, covMatCol);
      //不偏分散を利用する
      return divide(dot(b, a), data.size()[0]-1);
    })
  });

  return matrix(array2d);
}


//[0]が0次元目のデータの平均、[1]が1次元目のデータの平均と続く
export function means(data: Matrix): Matrix {
  const vec = generateSerialNumbers(data.size()[0]).reduce((mean, row) => {
    const vec = data.subset(index(row, generateSerialNumbers(data.size()[1])));
    return add(mean, vec) as Matrix;
  }, zeros([1, data.size()[1]]) as Matrix);

  const mean = dotDivide(vec, data.size()[0]) as Matrix;
  return mean;
}


//dataの各行に対して平均ベクトルとの差分を取る。共分散行列の計算に利用する。
function deviationMat(data: Matrix, means: Matrix): Matrix {
  return map(data, (value, i, _) => {
    const dimension = i[1];
    return subtract(value, means.subset(index(0, dimension)));
  });
}


export function getRow(m: Matrix, rowIndex: number): Matrix {
  return m.subset(index(rowIndex, generateSerialNumbers(m.size()[1])))
}


export function getCol(m: Matrix, colIndex: number): Matrix {
  return m.subset(index(generateSerialNumbers(m.size()[0]), colIndex))
}
