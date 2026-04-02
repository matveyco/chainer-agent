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

function computeRecentRuntimeHealth(matches = [], counters = {}) {
  const combatSignalRatio = computeCombatSignalRatio(matches);
  const avgFillRatio = matches.length
    ? matches.reduce((sum, match) => sum + Number(match?.fillRatio || 0), 0) / matches.length
    : 0;
  const avgShotRate = matches.length
    ? matches.reduce((sum, match) => {
      const decisions = Math.max(Number(match?.totalDecisionsMade || 0), 1);
      return sum + (Number(match?.totalShotsFired || 0) / decisions);
    }, 0) / matches.length
    : 0;
  const avgPolicyShare = matches.length
    ? matches.reduce((sum, match) => {
      const decisions = Math.max(Number(match?.totalDecisionsMade || 0), 1);
      const overrides = Number(match?.totalTacticalOverrides || 0);
      return sum + Math.max(0, Math.min(1, (decisions - overrides) / decisions));
    }, 0) / matches.length
    : 0;
  const avgDamagePerShot = matches.length
    ? matches.reduce((sum, match) => {
      const shots = Math.max(Number(match?.totalShotsFired || 0), 1);
      return sum + (Number(match?.totalDamageDealt || 0) / shots);
    }, 0) / matches.length
    : 0;
  const joinAttempts = Number(counters.joinAttempts || 0);
  const joinSuccessRate = joinAttempts > 0 ? Number(counters.joinSuccesses || 0) / joinAttempts : 1;

  return {
    combatSignalRatio: +combatSignalRatio.toFixed(4),
    avgFillRatio: +avgFillRatio.toFixed(4),
    avgShotRate: +avgShotRate.toFixed(4),
    avgPolicyShare: +avgPolicyShare.toFixed(4),
    avgDamagePerShot: +avgDamagePerShot.toFixed(4),
    joinSuccessRate: +joinSuccessRate.toFixed(4),
  };
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

function isCandidateValidated(familyStatus) {
  const aliases = familyStatus?.aliases || {};
  const report = familyStatus?.last_eval_report || null;
  const candidateVersion = Number(aliases.candidate || 0);
  const reportVersion = Number(report?.candidate_version || report?.challenger_version || 0);
  return Boolean(report?.passed && candidateVersion > 0 && reportVersion === candidateVersion);
}

function resolveStableModelAlias(familyStatus, config = {}) {
  const candidateVersion = Number(familyStatus?.aliases?.candidate || 0);
  if (candidateVersion > 0 && isCandidateValidated(familyStatus)) {
    return "candidate";
  }
  return config.training?.stableFallbackAlias || "champion";
}

function latestStagedOrEvaluatedVersion(familyStatus) {
  const aliases = familyStatus?.aliases || {};
  const report = familyStatus?.last_eval_report || null;
  return Math.max(
    Number(aliases.challenger || 0),
    Number(report?.challenger_version || 0),
    Number(report?.candidate_version || 0),
    0
  );
}

function shouldStageChallenger({
  familyStatus,
  recentMatches = [],
  counters = {},
  config = {},
  hasCurrentJob = false,
  queuedJobs = 0,
}) {
  if (hasCurrentJob || queuedJobs > 0) return { ok: false, reason: "evaluation_busy" };
  const latestVersion = Number(familyStatus?.aliases?.latest || 0);
  if (latestVersion <= 0) return { ok: false, reason: "latest_unavailable" };

  const lastGovernedVersion = latestStagedOrEvaluatedVersion(familyStatus);
  const minVersionDelta = Number(config.evaluation?.autoStageMinVersionDelta || 1000);
  if (latestVersion - lastGovernedVersion < minVersionDelta) {
    return { ok: false, reason: "version_delta_too_small", metrics: { latestVersion, lastGovernedVersion } };
  }

  const requiredMatches = Number(config.evaluation?.stagingRecentMatches || 4);
  if (recentMatches.length < requiredMatches) {
    return { ok: false, reason: "insufficient_recent_matches" };
  }

  const metrics = computeRecentRuntimeHealth(recentMatches, counters);
  if (metrics.combatSignalRatio < Number(config.evaluation?.stagingMinCombatSignalRatio || 0.75)) {
    return { ok: false, reason: "combat_signal_low", metrics };
  }
  if (metrics.avgFillRatio < Number(config.evaluation?.stagingMinFillRatio || 0.95)) {
    return { ok: false, reason: "fill_ratio_low", metrics };
  }
  if (metrics.joinSuccessRate < Number(config.evaluation?.stagingMinJoinSuccessRate || 0.97)) {
    return { ok: false, reason: "join_success_low", metrics };
  }
  if (metrics.avgShotRate < Number(config.evaluation?.stagingMinShotRate || 0.03)) {
    return { ok: false, reason: "shot_rate_low", metrics };
  }
  if (metrics.avgPolicyShare < Number(config.evaluation?.stagingMinPolicyShare || 0.1)) {
    return { ok: false, reason: "policy_share_low", metrics };
  }
  if (metrics.avgDamagePerShot < Number(config.evaluation?.stagingMinDamagePerShot || 0.25)) {
    return { ok: false, reason: "damage_per_shot_low", metrics };
  }

  return {
    ok: true,
    reason: "healthy",
    metrics: {
      ...metrics,
      latestVersion,
      lastGovernedVersion,
    },
  };
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
  computeRecentRuntimeHealth,
  hasMeaningfulCombatSignal,
  isCandidateValidated,
  latestStagedOrEvaluatedVersion,
  resolveStableModelAlias,
  selectLatestPassedEvaluationVersion,
  selectSafeRecoveryVersion,
  shouldStageChallenger,
  shouldQueueAutomaticEvaluation,
};
