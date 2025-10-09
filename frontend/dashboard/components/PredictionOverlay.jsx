/**
 * Prediction Overlay Visualization
 * Author: Member 4
 */
import React from "react";
import usePredictionData from "../../hooks/usePredictionData";

export default function PredictionOverlay({ metric }) {
  const { predictions } = usePredictionData(metric);
  if (!predictions) return null;

  return (
    <g>
      {predictions.map((pred, i) => (
        <line
          key={i}
          x1={pred.x}
          y1={pred.y}
          x2={pred.x2}
          y2={pred.y2}
          stroke={pred.color || "rgba(0, 255, 0, 0.5)"}
          strokeWidth={pred.strokeWidth || 2}
        />
      ))}
    </g>
  );
}
