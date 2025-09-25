import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
import dipy.reconst.fwdti as fwdti
import os
import re
import sys
from pathlib import Path
import traceback

# Use the actual path to your data
BASE_INPUT_DIRECTORY = sys.argv[1]  # This should be: /mnt/HPStorage/BlossomFW/fwtobeprocessed/dtifit_processed
OUTPUT_DIRECTORY = sys.argv[2]
BATCH_NUMBER = sys.argv[3]

# Create output directories
paths = [
    f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/FREE_WATER', 
    f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI', 
    f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/IMAGES'
]

for path in paths:
    Path(path).mkdir(parents=True, exist_ok=True)

subject_pattern = re.compile(r"sub-[a-zA-Z0-9]+")
session_pattern = re.compile(r"ses-[a-zA-Z0-9]+")

# Debug: Print the input directory to verify
print(f"Searching in: {BASE_INPUT_DIRECTORY}")

processed_count = 0

for dirpath, dirnames, filenames in os.walk(BASE_INPUT_DIRECTORY):
    if '/dwi' in dirpath:
        subject_match = subject_pattern.search(dirpath)
        session_match = session_pattern.search(dirpath)

        if subject_match and session_match:
            SUBJECT_ID = subject_match.group()
            SESSION_ID = session_match.group()

            print(f"Processing: {SUBJECT_ID} {SESSION_ID}")
            
            try:
                # Find the actual files in the directory
                dwi_files = [f for f in filenames if f.endswith('.nii.gz') and 'preproc_dwi' in f]
                bval_files = [f for f in filenames if f.endswith('.bval')]
                bvec_files = [f for f in filenames if f.endswith('.bvec')]
                mask_files = [f for f in filenames if f.endswith('.nii.gz') and 'mask' in f]
                
                if not (dwi_files and bval_files and bvec_files and mask_files):
                    print(f"  Missing required files in {dirpath}")
                    continue
                
                # Use the actual file names found
                diffusion_mri_file_path = os.path.join(dirpath, dwi_files[0])
                bval_file_path = os.path.join(dirpath, bval_files[0])
                bvec_file_path = os.path.join(dirpath, bvec_files[0])
                brain_mask_path = os.path.join(dirpath, mask_files[0])
                
                print(f"  DWI: {dwi_files[0]}")
                print(f"  BVAL: {bval_files[0]}")
                print(f"  BVEC: {bvec_files[0]}")
                print(f"  MASK: {mask_files[0]}")

                # Load data
                bvals, bvecs = read_bvals_bvecs(bval_file_path, bvec_file_path)
                gtab = gradient_table(bvals=bvals, bvecs=bvecs)

                diffusion_mri_image = nib.load(diffusion_mri_file_path)
                brain_mask_image = nib.load(brain_mask_path)
                diffusion_mri_data = diffusion_mri_image.get_fdata()
                brain_mask = brain_mask_image.get_fdata()

                # Process DTI
                print("  Running tensor model...")
                tensor_model = TensorModel(gtab)
                tensor_fit = tensor_model.fit(diffusion_mri_data, mask=brain_mask)
                
                print("  Running free water DTI model...")
                fw_dti_model = fwdti.FreeWaterTensorModel(gtab)
                fw_dti_fit = fw_dti_model.fit(diffusion_mri_data, mask=brain_mask)

                # Extract metrics
                dti_fa = tensor_fit.fa
                dti_ad = tensor_fit.ad
                dti_md = tensor_fit.md
                dti_rd = tensor_fit.rd

                affine = diffusion_mri_image.affine

                # Save DTI outputs
                nib.save(nib.Nifti1Image(np.nan_to_num(dti_ad), affine), 
                        f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_ad_map.nii.gz')
                nib.save(nib.Nifti1Image(np.nan_to_num(dti_md), affine), 
                        f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_md_map.nii.gz')
                nib.save(nib.Nifti1Image(np.nan_to_num(dti_rd), affine), 
                        f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_rd_map.nii.gz')
                nib.save(nib.Nifti1Image(np.nan_to_num(dti_fa), affine), 
                        f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_fa_map.nii.gz')

                # Save free water outputs
                fw_dti_fa = fw_dti_fit.fa
                fw_dti_md = fw_dti_fit.md
                nib.save(nib.Nifti1Image(np.nan_to_num(fw_dti_fa), affine), 
                        f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/FREE_WATER/{SUBJECT_ID}_{SESSION_ID}_fw_dti_fa_map.nii.gz')
                nib.save(nib.Nifti1Image(np.nan_to_num(fw_dti_md), affine), 
                        f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/FREE_WATER/{SUBJECT_ID}_{SESSION_ID}_fw_dti_md_map.nii.gz')

                # Create and save plots
                axial_slice = diffusion_mri_data.shape[2] // 2
                
                # First figure
                fig1, ax = plt.subplots(2, 4, figsize=(12, 6))
                for a in ax.flat:
                    a.set_xticks([])
                    a.set_yticks([])
                
                ax[0,0].imshow(fw_dti_fa[:, :, axial_slice].T, cmap='gray', origin='lower')
                ax[0,0].set_title('A) fwDTI FA')
                ax[0,1].imshow(dti_fa[:, :, axial_slice].T, cmap='gray', origin='lower')
                ax[0,1].set_title('B) standard DTI FA')
                
                fa_diff = abs(fw_dti_fa[:, :, axial_slice] - dti_fa[:, :, axial_slice])
                ax[0,2].imshow(fa_diff.T, cmap='hot', origin='lower')
                ax[0,2].set_title('C) FA difference')
                ax[0,3].axis('off')
                
                ax[1,0].imshow(fw_dti_md[:, :, axial_slice].T, cmap='gray', origin='lower')
                ax[1,0].set_title('D) fwDTI MD')
                ax[1,1].imshow(dti_md[:, :, axial_slice].T, cmap='gray', origin='lower')
                ax[1,1].set_title('E) standard DTI MD')
                
                md_diff = abs(fw_dti_md[:, :, axial_slice] - dti_md[:, :, axial_slice])
                ax[1,2].imshow(md_diff.T, cmap='hot', origin='lower')
                ax[1,2].set_title('F) MD difference')
                
                F = fw_dti_fit.f
                ax[1,3].imshow(F[:, :, axial_slice].T, cmap='viridis', origin='lower')
                ax[1,3].set_title('G) free water volume')
                
                plt.tight_layout()
                fig1.savefig(f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/IMAGES/{SUBJECT_ID}_{SESSION_ID}_In_vivo_free_water_DTI_and_standard_DTI_measures.png')
                plt.close(fig1)

                processed_count += 1
                print(f"  Successfully processed {SUBJECT_ID}_{SESSION_ID}")
                
            except Exception as e:
                print(f"  Error processing {SUBJECT_ID}_{SESSION_ID}: {str(e)}")
                print(traceback.format_exc())

print(f"Processing complete. Processed {processed_count} subjects.")
