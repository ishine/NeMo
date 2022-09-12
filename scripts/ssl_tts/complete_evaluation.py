import evaluate_synthesizer

ssl_model_ckpt_path = "/home/pneekhara/NeMo2022/SSLCheckPoints/SSLConformer22050_Epoch37.ckpt"
# ssl_model_ckpt_path = "/home/pneekhara/NeMo2022/tensorboards/ConformerModels/ConformerCompatibleTry3/ConformerCompatible_Epoch3.ckpt"
hifi_ckpt_path = "/home/pneekhara/NeMo2022/HiFiCKPTS/hifigan_libritts/HiFiLibriEpoch334.ckpt"

fastpitch_model_ckpts = [
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/DurationPredictor/SegMeanEpoch404.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/SpeakerLossFTTry3/SpeakerLossEpoch84.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/DurationPredictor/SegDurPerSampleEpoch604.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/LibriAllTraining/LibriAllFREpoch39.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/SpeakerLossFTTry3/SpeakerLossEpoch174.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/DurationPredictor/DurEpoch404.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/SpeakerLossFinetuning/SpeakerLossFTEpoch219.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitchLibriAll/LibriAllDataIDEpoch94.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitchLibriAll/LibriAllSingleDataEpoch89.ckpt",
    #
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/DurationPredictor/SegMeanEpoch604.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/DurationPredictor/SegDurInterp674.ckpt",
    # "/home/pneekhara/NeMo2022/tensorboards/FastPitch/DurationPredictor/SegDurPerSampleEpoch604.ckpt",
    #
    "/home/pneekhara/NeMo2022/tensorboards/FastPitch/DurationPredictor/LibriAllFirstRun149.ckpt",
]

manifest_paths = [
    "/home/pneekhara/Datasets/LibriDev/libri_dev_clean_local.json",
    "/home/pneekhara/NeMo2022/libri_train_formatted.json",
]
# manifest_paths = ["/home/pneekhara/Datasets/vctk/vctk_test_local.json"]
pitch_stats_jsons = [None, None]
# pitch_stats_jsons = [None]
# sv_model_names = ["speakerverification_speakernet", "ecapa_tdnn"]
sv_model_names = ["speakerverification_speakernet"]
evaluation_types = ["reconstructed", "swapping"]
base_out_dir = "/home/pneekhara/NeMo2022/Evaluations/ForPresentation"
n_speakers = 10
min_samples_per_spk = 15
max_samples_per_spk = 15
# precomputed_stats_fps = ["/home/pneekhara/NeMo2022/Evaluations/UpdatedEvaluation/stats_seen.pkl", "/home/pneekhara/NeMo2022/Evaluations/UpdatedEvaluation/stats_unseen.pkl"]
precomputed_stats_fps = [None, None]
compute_pitch = 1
compute_duration = 0
use_unique_tokens = 0
durations_per_speaker = [
    # 5,
    10,
    # 30
]

dataset_id = 1

for fastpitch_model_ckpt in fastpitch_model_ckpts:
    for midx, manifest_path in enumerate(manifest_paths):
        for evaluation_type in evaluation_types:
            for sv_model_name in sv_model_names:
                for duration_per_speaker in durations_per_speaker:
                    pitch_stats_json = pitch_stats_jsons[midx]
                    precomputed_stats_fp = precomputed_stats_fps[midx]
                    print(
                        "Evaluating",
                        fastpitch_model_ckpt,
                        manifest_path,
                        evaluation_type,
                        sv_model_name,
                        duration_per_speaker,
                    )
                    evaluate_synthesizer.evaluate(
                        manifest_path=manifest_path,
                        fastpitch_ckpt_path=fastpitch_model_ckpt,
                        ssl_model_ckpt_path=ssl_model_ckpt_path,
                        hifi_ckpt_path=hifi_ckpt_path,
                        sv_model_name=sv_model_name,
                        base_out_dir=base_out_dir,
                        n_speakers=n_speakers,
                        min_samples_per_spk=min_samples_per_spk,
                        max_samples_per_spk=max_samples_per_spk,
                        evaluation_type=evaluation_type,
                        pitch_stats_json=pitch_stats_json,
                        precomputed_stats_fp=precomputed_stats_fp,
                        compute_pitch=compute_pitch,
                        compute_duration=compute_duration,
                        use_unique_tokens=use_unique_tokens,
                        duration_per_speaker=duration_per_speaker,
                        dataset_id=dataset_id,
                    )