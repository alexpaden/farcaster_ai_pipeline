-- General unprocessed casts index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_status_0_initial 
  ON farcaster.casts ("timestamp")
  WHERE threads_status = 0;

-- (Optional but tiny) keep only the “label_value = 2” rows hot
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_labels_lbl2
  ON farcaster.user_labels (target_fid)
  WHERE label_value = '2';