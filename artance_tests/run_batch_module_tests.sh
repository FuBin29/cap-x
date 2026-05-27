#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

EPISODE_FILE="${EPISODE_FILE:-episodes.txt}"
TASKS="${TASKS:-close_drawer close_fridge close_microwave push_button toilet_seat_down}"
EPISODES="${EPISODES:-all}"
MODULES="${MODULES:-contact_point sam3 sam3_point_selection pointcloud}"
# Optional downstream modules: part_adjacency_plane end_effector_motion
DRY_RUN="${DRY_RUN:-0}"

VLM_MODEL="${VLM_MODEL:-gemini-2.5-pro}"
VLM_SERVER_URL="${VLM_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
SAM3_SERVICE_URL="${SAM3_SERVICE_URL:-http://127.0.0.1:8114}"
TOP_K="${TOP_K:-5}"
SUBSAMPLE_FACTOR="${SUBSAMPLE_FACTOR:-1}"

joint_type_for_task() {
  local task="$1"
  case "$task" in
    close_drawer|push_button)
      echo "prismatic"
      ;;
    close_fridge|close_microwave|toilet_seat_down)
      echo "revolute"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

require_prismatic_joint_module() {
  local module="$1"
  local task="$2"
  local joint_type

  joint_type="$(joint_type_for_task "$task")"
  if [[ "$joint_type" == "prismatic" ]]; then
    return 0
  fi

  echo "NotImplement: $module for $task uses joint_type=$joint_type; revolute-joint API is not configured yet." >&2
  return 1
}

contains_word() {
  local needle="$1"
  local haystack="$2"
  [[ " ${haystack} " == *" ${needle} "* ]]
}

selected_episode() {
  local task="$1"
  local variation="$2"
  local episode="$3"
  local frame="$4"

  contains_word "$task" "$TASKS" || return 1
  [[ "$EPISODES" == "all" ]] && return 0

  contains_word "$task:$variation:$episode:$frame" "$EPISODES" && return 0
  contains_word "$task:$variation:$episode" "$EPISODES" && return 0
  contains_word "$task" "$EPISODES" && return 0
  return 1
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

delete_existing_module_output() {
  local module="$1"
  local task="$2"
  local variation="$3"
  local episode="$4"
  local frame="$5"
  local camera="$6"
  local padded_frame
  local episode_key
  local output_dir

  printf -v padded_frame '%03d' "$frame"
  episode_key="variation${variation}_episode${episode}_frame${padded_frame}_${camera}"
  output_dir="$task/outputs/$module/$episode_key"

  if [[ -e "$output_dir" ]]; then
    printf '+ rm -rf %q\n' "$output_dir"
    if [[ "$DRY_RUN" != "1" ]]; then
      rm -rf -- "$output_dir"
    fi
  fi
}

run_module_for_episode() {
  local module="$1"
  local task="$2"
  local rgb_path="$3"
  local variation="$4"
  local episode="$5"
  local frame="$6"
  local camera="$7"
  local padded_frame
  local episode_key

  printf -v padded_frame '%03d' "$frame"
  episode_key="variation${variation}_episode${episode}_frame${padded_frame}_${camera}"

  delete_existing_module_output "$module" "$task" "$variation" "$episode" "$frame" "$camera"

  case "$module" in
    contact_point)
      run_cmd uv run --no-sync --active python run_task_module.py contact_point \
        --task "$task" \
        --episode-line "$rgb_path" \
        --model "$VLM_MODEL" \
        --server-url "$VLM_SERVER_URL"
      ;;
    sam3)
      run_cmd uv run --no-sync --active python run_task_module.py sam3 \
        --task "$task" \
        --episode-line "$rgb_path" \
        --service-url "$SAM3_SERVICE_URL" \
        --top-k "$TOP_K" \
        --no-show
      ;;
    sam3_point_selection)
      run_cmd uv run --no-sync --active python run_task_module.py sam3_point_selection \
        --task "$task" \
        --episode-line "$rgb_path" \
        --service-url "$SAM3_SERVICE_URL" \
        --top-k "$TOP_K" \
        --no-show
      ;;
    pointcloud)
      run_cmd uv run --no-sync --active python run_task_module.py pointcloud \
        --task "$task" \
        --episode-line "$rgb_path" \
        --subsample-factor "$SUBSAMPLE_FACTOR" \
        --mask-only \
        --mask-overlap-policy first-wins
      ;;
    part_adjacency_plane)
      local pointcloud_npz
      local pointcloud_summary
      require_prismatic_joint_module "$module" "$task" || return 0
      pointcloud_npz="$task/outputs/pointcloud/${episode_key}/${frame}/${frame}_camera_sam3_parts.npz"
      pointcloud_summary="$task/outputs/pointcloud/${episode_key}/${frame}/pointcloud_summary.json"
      if [[ ! -f "$task/part_adjacency_plane_test.py" ]]; then
        echo "Skipping part_adjacency_plane for $task: missing $task/part_adjacency_plane_test.py" >&2
        return 0
      fi
      run_cmd uv run --no-sync --active python "$task/part_adjacency_plane_test.py" \
        --pointcloud-npz "$pointcloud_npz" \
        --source-summary "$pointcloud_summary" \
        --output-dir "$task/outputs/part_adjacency_plane"
      ;;
    end_effector_motion)
      local plane_summary
      require_prismatic_joint_module "$module" "$task" || return 0
      plane_summary="$task/outputs/part_adjacency_plane/${episode_key}/${frame}/part_adjacency_plane_summary.json"
      if [[ ! -f "$task/end_effector_motion_test.py" ]]; then
        echo "Skipping end_effector_motion for $task: missing $task/end_effector_motion_test.py" >&2
        return 0
      fi
      run_cmd uv run --no-sync --active python "$task/end_effector_motion_test.py" \
        --plane-summary "$plane_summary" \
        --output-dir "$task/outputs/end_effector_motion/${episode_key}" \
        --camera-name "$camera"
      ;;
    *)
      echo "Unsupported module: $module" >&2
      exit 2
      ;;
  esac
}

while IFS= read -r rgb_path; do
  [[ -z "$rgb_path" || "$rgb_path" == \#* ]] && continue

  if [[ "$rgb_path" =~ /RLBench-data/([^/]+)/variation([0-9]+)/episodes/episode([0-9]+)/([^/]+)_rgb/([0-9]+)\.png$ ]]; then
    task="${BASH_REMATCH[1]}"
    variation="${BASH_REMATCH[2]}"
    episode="${BASH_REMATCH[3]}"
    camera="${BASH_REMATCH[4]}"
    frame="${BASH_REMATCH[5]}"
  else
    echo "Skipping unrecognized episode path: $rgb_path" >&2
    continue
  fi

  selected_episode "$task" "$variation" "$episode" "$frame" || continue

  for module in $MODULES; do
    run_module_for_episode "$module" "$task" "$rgb_path" "$variation" "$episode" "$frame" "$camera"
  done
done < "$EPISODE_FILE"
