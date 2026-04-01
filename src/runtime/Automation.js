function hasMeaningfulCombatSignal(match) {
  if (!match || typeof match !== "object") return false;
  return Boolean(
    match.hasCombatSignal ||
    (match.totalScore || 0) > 0 ||
    (match.totalDamageDealt || 0) > 0 ||
    (match.totalKills || 0) > 0 ||
    (match.totalDeaths || 0) > 0
  );
}

function computeCombatSignalRatio(matches = []) {
  if (!Array.isArray(matches) || matches.length === 0) return 0;
  const signaled = matches.filter((match) => hasMeaningfulCombatSignal(match)).length;
  return signaled / matches.length;
}

function selectLatestPassedEvaluationVersion(familyStatus) {
  const history = Array.isArray(familyStatus?.evaluation_history) ? familyStatus.evaluation_history : [];
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const report = history[index];
    const version = Number(report?.candidate_version || 0);
    if (report?.passed && Number.isFinite(version) && version > 0) {
      return version;
    }
  }
  return 0;
}

function selectSafeRecoveryVersion(familyStatus) {
  const passed = selectLatestPassedEvaluationVersion(familyStatus);
  if (passed > 0) return passed;
  const champion = Number(familyStatus?.aliases?.champion || 0);
  return Number.isFinite(champion) && champion > 0 ? champion : 0;
}

function shouldQueueAutomaticEvaluation({
  totalMatches = 0,
  selectionInterval = 10,
  lastQueuedMatchCount = 0,
  latestVersion = 0,
  candidateVersion = 0,
  hasCurrentJob = false,
  queuedJobs = 0,
}) {
  if (hasCurrentJob || queuedJobs > 0) return false;
  if (!Number.isFinite(totalMatches) || totalMatches <= 0) return false;
  if (!Number.isFinite(selectionInterval) || selectionInterval <= 0) return false;
  if (totalMatches % selectionInterval !== 0) return false;
  if (lastQueuedMatchCount === totalMatches) return false;
  if (!Number.isFinite(latestVersion) || latestVersion <= 0) return false;
  if (!Number.isFinite(candidateVersion) || candidateVersion < 0) return false;
  return latestVersion > candidateVersion;
}

module.exports = {
  computeCombatSignalRatio,
  hasMeaningfulCombatSignal,
  selectLatestPassedEvaluationVersion,
  selectSafeRecoveryVersion,
  shouldQueueAutomaticEvaluation,
};
