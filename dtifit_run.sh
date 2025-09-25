#!/bin/bash

# Input and output base directories
INPUT_DIR="/mnt/HPStorage/BlossomFW/fwtobeprocessed"
OUTPUT_DIR="/mnt/HPStorage/BlossomFW/fwtobeprocessed/dtifit_processed"

# Loop through subjects
for subj in $INPUT_DIR/sub-*; do
    subj_id=$(basename $subj)
    
    for ses in $subj/ses-*; do
        ses_id=$(basename $ses)
        dwi_dir="$ses/dwi"

        echo "Processing $subj_id $ses_id ..."

        # Make output directory
        out_dir="$OUTPUT_DIR/$subj_id/$ses_id/dwi"
        mkdir -p "$out_dir"

        # Get input files (only acq-80dir)
        dwi=$(ls $dwi_dir/*acq-80dir*_desc-preproc_dwi.nii.gz 2>/dev/null | head -n 1)
        bval=${dwi%.nii.gz}.bval
        bvec=${dwi%.nii.gz}.bvec

        if [[ -z $dwi ]]; then
            echo "⚠️ No 80dir DWI found for $subj_id $ses_id. Skipping..."
            continue
        fi

        # Brain mask (if available from qsiprep, restricted to 80dir as well)
        mask=$(ls $dwi_dir/*acq-80dir*mask.nii.gz 2>/dev/null | head -n 1)

        if [[ -z $mask ]]; then
            echo "No mask found, generating with bet ..."
            fslroi $dwi "$out_dir/b0" 0 1
            bet "$out_dir/b0" "$out_dir/b0_brain" -m -f 0.3
            mask="$out_dir/b0_brain_mask.nii.gz"
        fi

        # Run dtifit
        dtifit \
            -k $dwi \
            -o "$out_dir/${subj_id}_${ses_id}_dtifit" \
            -m $mask \
            -r $bvec \
            -b $bval

        # Copy DWI, bval, bvec, and mask for free-water analysis
        cp $dwi $out_dir/
        cp $bval $out_dir/
        cp $bvec $out_dir/
        cp $mask $out_dir/
        
        echo "✅ Done with $subj_id $ses_id"
    done
done

