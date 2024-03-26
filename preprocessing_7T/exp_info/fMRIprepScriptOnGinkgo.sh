export FMRIPREP=$NEUROSPIN_DIR/unicog/resources/softwares/FMRIPREP/fmriprep-22.0.2.simg
export PROJECT=$NEUROSPIN_DIR/icortex/iCortexDatabase/06-fMRI
export FREESURFER=$FREESURFER_HOME
export SOURCE=03-BidsConversion
export DEST=04-fMRIPrep
singularity run --cleanenv \
            -B "${PROJECT}:/data" \
            -B "${FREESURFER}:/freesurfer" \
            ${FMRIPREP} \
            "/data/${SOURCE}" \
            "/data/$DEST" \
            participant \
                --participant-label $1 \
                --output-spaces T1w \
                --bold2t1w-init header \
                --bold2t1w-dof 6 \
                --clean-workdir \
                --fs-license-file "/freesurfer/license.txt" \
                -w "/data/${DEST}/work_dir"
