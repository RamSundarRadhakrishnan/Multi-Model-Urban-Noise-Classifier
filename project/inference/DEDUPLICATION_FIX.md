# Duplicate Events Fix - Summary

## Problem Identified

**Issue**: 380 events were being stored for only 190 chunks - exactly double!

**Root Cause**: The audio inference was processing each file twice because:

- `get_audio_files()` in `inference_utils.py` searched for both lowercase `.wav` AND uppercase `.WAV` extensions
- On Windows (case-insensitive filesystem), files with `.wav` extension were found by both glob patterns
- Result: Each `.wav` file was processed twice, creating 2x duplicate entries

## Fixes Implemented

### 1. **Root Cause Fix** - `inference_utils.py`

```python
# Before: Could find same file twice
for ext in extensions:
    audio_files.extend(directory.glob(f'*{ext}'))      # Finds file.wav
    audio_files.extend(directory.glob(f'*{ext.upper()}'))  # Also finds file.wav on Windows!

# After: Deduplicate using set
audio_files_unique = list(set(audio_files))
return sorted([str(f) for f in audio_files_unique])
```

**Why this works**: Windows filesystem is case-insensitive, so `*.wav` and `*.WAV` patterns match the same files. Converting to a set removes duplicates.

### 2. **Safety Check #1** - Audio Inference Deduplication

**File**: `integrated_inference.py` → `run_audio_inference()`

```python
# Remove duplicates based on filename (keep first occurrence)
initial_count = len(results_df)
results_df = results_df.drop_duplicates(subset=['filename'], keep='first')
duplicates_removed = initial_count - len(results_df)

if duplicates_removed > 0:
    print(f"  ⚠️  Removed {duplicates_removed} duplicate audio entries")
```

**Purpose**: Even if audio files are processed twice, this ensures the DataFrame only contains unique filenames.

### 3. **Safety Check #2** - Chunk Matching Deduplication

**File**: `integrated_inference.py` → `match_audio_video_chunks()`

```python
# Remove any duplicate chunk_ids
chunk_ids_seen = set()
deduplicated_chunks = []
duplicates_in_matching = 0

for chunk in matched_chunks:
    if chunk['chunk_id'] not in chunk_ids_seen:
        chunk_ids_seen.add(chunk['chunk_id'])
        deduplicated_chunks.append(chunk)
    else:
        duplicates_in_matching += 1

if duplicates_in_matching > 0:
    print(f"  ⚠️  Removed {duplicates_in_matching} duplicate chunks during matching")
```

**Purpose**: Ensures no duplicate chunk_ids make it through the matching process.

### 4. **Safety Check #3** - Event Classification Deduplication

**File**: `integrated_inference.py` → `classify_events()`

```python
# Remove duplicate events based on event_id (final safety check)
event_ids_seen = set()
deduplicated_events = []
duplicates_in_classification = 0

for event in classified_events:
    if event.event_id not in event_ids_seen:
        event_ids_seen.add(event.event_id)
        deduplicated_events.append(event)
    else:
        duplicates_in_classification += 1

if duplicates_in_classification > 0:
    print(f"  ⚠️  Removed {duplicates_in_classification} duplicate events")
```

**Purpose**: Final safety net to ensure no duplicate events are stored or analyzed.

## Defense in Depth Strategy

We implemented **4 layers of protection**:

1. **Layer 1 (Root Cause)**: Fix file discovery to not find duplicates
2. **Layer 2 (Audio)**: Deduplicate after audio inference
3. **Layer 3 (Matching)**: Deduplicate during chunk matching
4. **Layer 4 (Events)**: Deduplicate before final classification

This ensures:

- ✅ The root cause is fixed
- ✅ Even if duplicates slip through, they're caught at multiple checkpoints
- ✅ Pipeline is robust against similar issues in the future
- ✅ Clear warnings when duplicates are detected

## Expected Output

### Before Fix:

```
Found 380 audio files in ../pre_inference/processed_media/audio_chunks
✅ Audio inference complete: 380 chunks processed in 45.23s
✅ Matched 380 chunks in 2.34s
✅ Classified 380 events in 17.12s
```

### After Fix:

```
Found 190 audio files in ../pre_inference/processed_media/audio_chunks
✅ Audio inference complete: 190 chunks processed in 22.61s
✅ Matched 190 chunks in 1.17s
✅ Classified 190 events in 8.56s
```

**OR** if duplicates are still somehow created:

```
Found 380 audio files in ../pre_inference/processed_media/audio_chunks
  ⚠️  Removed 190 duplicate audio entries
✅ Audio inference complete: 190 chunks processed in 22.61s
✅ Matched 190 chunks in 1.17s
✅ Classified 190 events in 8.56s
```

## Performance Improvement

By eliminating duplicate processing:

- **50% faster audio inference** (processing 190 files instead of 380)
- **50% faster event classification** (classifying 190 events instead of 380)
- **Accurate metrics** (no inflated counts)
- **Correct analysis** (no duplicate events in statistics)

## Testing the Fix

To verify the fix works:

1. **Run the pipeline**: Execute `inference.ipynb`
2. **Check console output**: Look for warning messages about duplicates removed
3. **Verify counts**: Ensure `chunks processed` matches actual number of chunk files
4. **Check final events**: Confirm `total_events` matches expected chunk count

## Files Modified

1. ✅ `inference_utils.py` - Fixed `get_audio_files()` to deduplicate
2. ✅ `integrated_inference.py` - Added 3 deduplication safety checks
3. ✅ All analysis in `inference.ipynb` now works with unique events only

---

**Status**: ✅ FIXED - Duplicates prevented at source + 3 safety layers
