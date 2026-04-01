export type ScoreInput = {
  head_position: number;
  shoulders_symmetry: number;
  thoracic_curvature: number;
  hip_alignment: number;
  knee_alignment: number;
  feet_positioning: number;
  ankle_dorsiflexion: number;
  hip_rotation: number;
  shoulder_flexion: number;
  thoracic_rotation: number;
  plank_time: number;
  single_leg_balance: number;
  squat_pattern: number;
  push_pull_quality: number;
  pain_scale: number;
};

export function computeMovementQualityScore(values: ScoreInput) {
  const positive = [
    values.head_position,
    values.shoulders_symmetry,
    values.thoracic_curvature,
    values.hip_alignment,
    values.knee_alignment,
    values.feet_positioning,
    values.ankle_dorsiflexion,
    values.hip_rotation,
    values.shoulder_flexion,
    values.thoracic_rotation,
    values.plank_time,
    values.single_leg_balance,
    values.squat_pattern,
    values.push_pull_quality,
  ];

  const max = positive.length * 3;
  const adjustedPain = 3 - Math.min(3, Math.round((values.pain_scale / 10) * 3));
  const total = positive.reduce((a, b) => a + b, 0) + adjustedPain;
  const score = (total / (max + 3)) * 100;
  const riskFlag = values.pain_scale >= 7 || score < 45;

  return { score: Number(score.toFixed(1)), riskFlag: riskFlag ? 1 : 0 };
}
