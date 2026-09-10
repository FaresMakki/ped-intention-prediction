"""Published test-set evidence and descriptions for comparison methods."""

from __future__ import annotations

from typing import Any

METHOD_CATALOG: dict[str, dict[str, Any]] = {
    "trajectory_social": {
        "name": "Trajectory + Social",
        "short_name": "Traj + Social",
        "architecture": "Locked classwise Rule B fusion",
        "role": "Final selected system",
        "selection_note": "Validation-selected weights; current deployment model.",
        "is_final": True,
        "metrics": {
            "binary_macro_f1": 0.8761737992610837,
            "crossing_recall": 0.8146341463414634,
            "crossing_f1": 0.8226600985221675,
            "three_class_macro_f1": 0.7266499336580714,
            "pr_auc": 0.8377163353278604,
        },
    },
    "environment": {
        "name": "Environment Graph",
        "short_name": "Environment",
        "architecture": "12-node attention environment graph",
        "role": "Standalone branch",
        "selection_note": "Epoch 10 selected on validation mapped macro F1.",
        "is_final": False,
        "metrics": {
            "binary_macro_f1": 0.816395496939069,
            "crossing_recall": 0.6487804878048781,
            "crossing_f1": 0.726775956284153,
            "three_class_macro_f1": 0.7441463238427186,
            "pr_auc": 0.7427773722959999,
        },
    },
    "skeleton": {
        "name": "Skeleton Graph",
        "short_name": "Skeleton",
        "architecture": "Adaptive 19-joint PedGraphPlus GCN",
        "role": "Standalone branch",
        "selection_note": "Epoch 61 selected on validation mapped macro F1.",
        "is_final": False,
        "metrics": {
            "binary_macro_f1": 0.6143540669856459,
            "crossing_recall": 0.4097560975609756,
            "crossing_f1": 0.43636363636363634,
            "three_class_macro_f1": 0.4412726947291418,
            "pr_auc": 0.43064467533674916,
        },
    },
    "environment_skeleton": {
        "name": "Environment + Skeleton",
        "short_name": "Env + Skeleton",
        "architecture": "Uncertainty-gated residual logit correction",
        "role": "Validation-selected fusion variant",
        "selection_note": (
            "Epoch 7 selected on validation; test-only variants are excluded."
        ),
        "is_final": False,
        "metrics": {
            "binary_macro_f1": 0.8148907646601953,
            "crossing_recall": 0.6487804878048781,
            "crossing_f1": 0.7247956403269755,
            "three_class_macro_f1": 0.7438513330876674,
            "pr_auc": 0.7475042675988496,
        },
    },
    "relation_risk": {
        "name": "Relation-Risk Graph",
        "short_name": "Relation-Risk",
        "architecture": "8-node relation-risk attention graph",
        "role": "Standalone branch",
        "selection_note": "Epoch 14 selected on validation mapped macro F1.",
        "is_final": False,
        "metrics": {
            "binary_macro_f1": 0.8233369485212841,
            "crossing_recall": 0.6439024390243903,
            "crossing_f1": 0.7353760445682451,
            "three_class_macro_f1": 0.7291998659137207,
            "pr_auc": 0.809456057515791,
        },
    },
}

METHOD_ORDER = (
    "trajectory_social",
    "relation_risk",
    "environment",
    "environment_skeleton",
    "skeleton",
)
