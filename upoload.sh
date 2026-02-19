#!/bin/bash
# 이름: upload.sh
set -euo pipefail

BRANCH="main"
REMOTE="origin"

# commit message: first arg or default
MSG="${1:-}"

echo "🔄 git pull ${REMOTE} ${BRANCH}"
git pull "${REMOTE}" "${BRANCH}"
echo "✅ 최신 상태로 업데이트 완료!"

echo "➕ git add ."
git add .

echo "📌 git status"
git status

# 변경사항 없으면 종료
if git diff --cached --quiet; then
  echo "🟨 스테이징된 변경사항이 없음. (커밋/푸시 생략)"
  exit 0
fi

# 메시지 없으면 기본 메시지 생성 (날짜 포함)
if [[ -z "${MSG}" ]]; then
  MSG="Update blog posts ($(date '+%Y-%m-%d %H:%M'))"
fi

echo "📝 git commit -m \"${MSG}\""
git commit -m "${MSG}"

echo "🚀 git push ${REMOTE} ${BRANCH}"
git push "${REMOTE}" "${BRANCH}"

echo "🎉 완료! (${BRANCH}에 push됨)"