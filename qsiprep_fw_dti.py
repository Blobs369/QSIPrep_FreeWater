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


BASE_INPUT_DIRECTORY = sys.argv[1]
OUTPUT_DIRECTORY = sys.argv[2]
BATCH_NUMBER = sys.argv[3]
INPUT_DIRECTORY = str(Path(BASE_INPUT_DIRECTORY) / f"BATCH-{BATCH_NUMBER}")


paths = [f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/FREE_WATER', f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI', f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/IMAGES']

for path in paths:
    Path(path).mkdir(parents=True, exist_ok=True)

subject_pattern = re.compile(r"sub-[a-zA-Z0-9]+")
session_pattern = re.compile(r"ses-[a-zA-Z0-9]+")

for dirpath, dirnames, filenames in os.walk(INPUT_DIRECTORY):
    if '/dwi' in dirpath:
        subject_match = subject_pattern.search(dirpath)
        session_match = session_pattern.search(dirpath)
        
        if subject_match and session_match:
            SUBJECT_ID = subject_match.group()
            SESSION_ID = session_match.group()

            print(SUBJECT_ID, SESSION_ID)

            diffusion_mri_file_path = f'{INPUT_DIRECTORY}/{SUBJECT_ID}/{SESSION_ID}/dwi/{SUBJECT_ID}_{SESSION_ID}_acq-80dir_run-01_space-ACPC_desc-preproc_dwi.nii.gz'
            bval_file_path = f'{INPUT_DIRECTORY}/{SUBJECT_ID}/{SESSION_ID}/dwi/{SUBJECT_ID}_{SESSION_ID}_acq-80dir_run-01_space-ACPC_desc-preproc_dwi.bval'
            bvec_file_path = f'{INPUT_DIRECTORY}/{SUBJECT_ID}/{SESSION_ID}/dwi/{SUBJECT_ID}_{SESSION_ID}_acq-80dir_run-01_space-ACPC_desc-preproc_dwi.bvec'
            brain_mask_path = f'{INPUT_DIRECTORY}/{SUBJECT_ID}/{SESSION_ID}/dwi/{SUBJECT_ID}_{SESSION_ID}_acq-80dir_run-01_space-ACPC_desc-brain_mask.nii.gz'

            bvals, bvecs = read_bvals_bvecs(bval_file_path, bvec_file_path)
            gtab = gradient_table(bvals=bvals, bvecs=bvecs)

            diffusion_mri_image = nib.load(diffusion_mri_file_path)
            brain_mask_image = nib.load(brain_mask_path)
            diffusion_mri_data = diffusion_mri_image.get_fdata()
            brain_mask = brain_mask_image.get_fdata()

            tensor_model = TensorModel(gtab)
            tensor_fit = tensor_model.fit(diffusion_mri_data, mask=brain_mask)
            fw_dti_model = fwdti.FreeWaterTensorModel(gtab)
            fw_dti_fit = fw_dti_model.fit(diffusion_mri_data, mask=brain_mask)

            dti_fa = tensor_fit.fa
            dti_ad = tensor_fit.ad
            dti_md = tensor_fit.md
            dti_rd = tensor_fit.rd

            affine = diffusion_mri_image.affine

            dti_ad_clean = np.nan_to_num(dti_ad)
            dti_ad_nifti = nib.Nifti1Image(dti_ad_clean, affine)
            nib.save(dti_ad_nifti, f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_ad_map.nii.gz')

            dti_md_clean = np.nan_to_num(dti_md)
            dti_md_nifti = nib.Nifti1Image(dti_md_clean, affine)
            nib.save(dti_md_nifti, f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_md_map.nii.gz')

            dti_rd_clean = np.nan_to_num(dti_rd)
            dti_rd_nifti = nib.Nifti1Image(dti_rd_clean, affine)
            nib.save(dti_rd_nifti, f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_rd_map.nii.gz')

            dti_fa_clean = np.nan_to_num(dti_fa)
            dti_fa_nifti = nib.Nifti1Image(dti_fa_clean, affine)
            nib.save(dti_fa_nifti, f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/DTI/{SUBJECT_ID}_{SESSION_ID}_dti_fa_map.nii.gz')

            fw_dti_fa = fw_dti_fit.fa
            fw_dti_md = fw_dti_fit.md
            fw_dti_fa_clean = np.nan_to_num(fw_dti_fa)
            fw_dti_md_clean = np.nan_to_num(fw_dti_md)
            fw_dti_fa_nifti = nib.Nifti1Image(fw_dti_fa_clean, affine)
            fw_dti_md_nifti = nib.Nifti1Image(fw_dti_md_clean, affine)

            nib.save(fw_dti_fa_nifti, f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/FREE_WATER/{SUBJECT_ID}_{SESSION_ID}_fw_dti_fa_map.nii.gz')
            nib.save(fw_dti_md_nifti, f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/FREE_WATER/{SUBJECT_ID}_{SESSION_ID}_fw_dti_md_map.nii.gz')

            axial_slice = diffusion_mri_data.shape[2] // 2
            fig1, ax = plt.subplots(2, 4, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
            fig1.subplots_adjust(hspace=0.3, wspace=0.05)

            ax.flat[0].set_title('A) fwDTI FA')
            ax.flat[1].set_title('B) standard DTI FA')
            fa_difference = abs(fw_dti_fa[:, :, axial_slice] - dti_fa[:, :, axial_slice])
            ax.flat[2].set_title('C) FA difference')
            ax.flat[3].axis('off')
            ax.flat[4].set_title('D) fwDTI MD')
            ax.flat[5].set_title('E) standard DTI MD')
            md_difference = abs(fw_dti_md[:, :, axial_slice] - dti_md[:, :, axial_slice])
            ax.flat[6].set_title('F) MD difference')

            F = fw_dti_fit.f
            ax.flat[7].set_title('G) free water volume')

            fw_dti_fa[F > 0.7] = 0
            dti_fa[F > 0.7] = 0

            fig2, ax = plt.subplots(1, 3, figsize=(9, 3), subplot_kw={'xticks': [], 'yticks': []})
            fig2.subplots_adjust(hspace=0.3, wspace=0.05)

            ax.flat[0].set_title('A) fwDTI FA')
            ax.flat[1].set_title('B) standard DTI FA')
            fa_difference = abs(fw_dti_fa[:, :, axial_slice] - dti_fa[:, :, axial_slice])
            ax.flat[2].set_title('C) FA difference')

            fig1.savefig(f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/IMAGES/{SUBJECT_ID}_{SESSION_ID}_In_vivo_free_water_DTI_and_standard_DTI_measures.png')
            fig2.savefig(f'{OUTPUT_DIRECTORY}/BATCH-{BATCH_NUMBER}/IMAGES/{SUBJECT_ID}_{SESSION_ID}_In_vivo_free_water_DTI_and_standard_DTI_corrected.png')

