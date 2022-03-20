import { Vector, add, division, sub, squaredNorm } from "./linearAlgebra";

export type Cluster<T extends number> = Vector<T>[];

export function Kmeans<T extends number>(data: Vector<T>[], k: number): {clusters: Cluster<T>[], means: Vector<T>[]} {
  const initIndices = Array(data.length).fill(null).map((_, i) => i % k);

  const init = data.reduce<Vector<T>[][]>((pre, current, i) => {
    pre[initIndices[i]].push(current);
    return pre;
  }, Array(k).fill(null).map(() => []));

  const means = calcMeanVectors(init);

  return KmeansRec(data, means);
}

export function SequentialKmeans<T extends number>(data: Vector<T>[], k: number): 
  { clusters: Cluster<T>[], means: Vector<T>[], next: ( means: Vector<T>[]) => {clusters: Cluster<T>[], means: Vector<T>[]}} {
    const initIndices = Array(data.length).fill(null).map((_, i) => i % k);

    const clusters = data.reduce<Vector<T>[][]>((pre, current, i) => {
      pre[initIndices[i]].push(current);
      return pre;
    }, Array(k).fill(null).map(() => []));
  
    const means = calcMeanVectors(clusters);

    const next = (means: Vector<T>[]): {clusters: Cluster<T>[], means: Vector<T>[]}  => {
      const newClusters = applyClustering(data, means);
      return {
        clusters: newClusters,
        means: calcMeanVectors(newClusters)
      }
    }

    return {clusters, means, next};
}

function KmeansRec<T extends number>(data: Vector<T>[], meanVectors: Vector<T>[]): {clusters: Cluster<T>[], means: Vector<T>[]} {
  const clusters = applyClustering(data, meanVectors);
  const means = calcMeanVectors(clusters);
  if (isSameVectors(meanVectors, means)) {
    return {clusters, means};
  }

  return KmeansRec(data, means);
}

function calcMeanVectors<T extends number>(clusters: Cluster<T>[]): Vector<T>[] {
  return clusters.map(cluster => {
    return division(
      cluster.reduce((pre, current, i) => i===0? current: add(current, pre), cluster[0])
    , cluster.length
    );
  });
}

function isSameVectors<T extends number>(vec1: Vector<T>[], vec2: Vector<T>[]): boolean {
  if (vec1.length !== vec2.length) {
    return false
  }

  return vec1.reduce<boolean>((pre, _, i) => {
    const delta = sub(vec1[i], vec2[i]);
    return (squaredNorm(delta) < 0.00001) && pre;
  }, true);
}

//vectorsの中からsrcに最も近い要素のインデックスを返す
function nearest<T extends number>(src: Vector<T>, vectors: Vector<T>[]): number {
  //squaredNorm(sub(src, vectors[0]))と書くと再帰が深すぎるといわれる
  const first = sub(src, vectors[0]);
  const initSolution = {
    index: 0,
    dist: squaredNorm(first),
  }

  return vectors.reduce<{index: number, dist: number}>((pre, current, i) => {
    if (i === 0) {
      return pre;
    }

    const delta = sub(src, current);
    const d = squaredNorm(delta);
    if (d < pre.dist) {
      return {
        index: i,
        dist: d
      }
    }

    return pre;
  }, initSolution).index;
}

function applyClustering<T extends number>(data: Vector<T>[], meanVectors: Vector<T>[]): Cluster<T>[] {
  const indicies = data.map(vec => nearest(vec, meanVectors));

  const newCluster: Cluster<T>[] = Array(meanVectors.length).fill(null).map(() => []);
  return indicies.reduce((pre, current, i) => {
    pre[current].push(data[i]);
    return pre;
  }, newCluster);
}