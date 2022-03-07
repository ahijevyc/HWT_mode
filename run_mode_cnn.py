import argparse
import yaml
from os.path import exists, join, isfile
from os import makedirs, path
import pandas as pd
import joblib
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, get_meta_scalars, predict_labels_gmm, \
    predict_labels_cnn, predict_labels_dnn, save_labels, merge_labels
from hwtmode.process import fetch_storm_reports, generate_obs_grid, generate_mode_grid
from hwtmode.models import load_conv_net
from hwtmode.evaluation import bss, brier_score
from tensorflow.keras.models import load_model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-e", "--eval", action="store_true", help="Evaluate conditional probabilities.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    models, gmms, neuron_activations, labels = {}, {}, {}, {}
    if config["run_start_date"] == "today":
        if config['run_freq'] == 'hourly':
            start_str = (pd.Timestamp(config["run_start_date"], tz="UTC") - pd.Timedelta(hours=3)).strftime("%Y%m%d-%H00")
            end_str = (pd.Timestamp(config["run_end_date"], tz="UTC") - pd.Timedelta(hours=3)).strftime("%Y%m%d-%H00")
        elif config['run_freq'] == 'daily':
            start_str = (pd.Timestamp(config["run_start_date"], tz="UTC")).strftime("%Y%m%d-0000")
            end_str = (pd.Timestamp(config["run_end_date"], tz="UTC")).strftime("%Y%m%d-0000")
    else:
        start_str = (pd.Timestamp(config["run_start_date"], tz="UTC")).strftime("%Y%m%d-%H00")
        end_str = (pd.Timestamp(config["run_end_date"], tz="UTC")).strftime("%Y%m%d-%H00")
    if start_str != end_str:
        date_str = start_str + '_' + end_str
    else:
        date_str = start_str

    l = []
    for d in pd.date_range(start_str.replace('-', ''), end_str.replace('-', ''), freq=config['run_freq'][0]):
        file_path = join(config["data_path"].replace('_nc/', '_csv/'),
                         f'{config["csv_model_prefix"]}{d.strftime("%Y%m%d-%H00")}.csv')
        if exists(file_path):
            df = pd.read_csv(file_path)
            l.append(df)

    storm_data = pd.concat(l).reset_index(drop=True)

    for model_type, model_dict in config["models"].items():
        for model_name in model_dict.keys():
            model_path = join(model_dict[model_name]["model_path"], "models", model_name)
            label_path = join(model_dict[model_name]["model_path"], 'labels')
            if not exists(label_path):
                makedirs(label_path)

            if model_type != 'supervised_DNN':
                scale_values = pd.read_csv(join(model_path, f"scale_values_{model_name}.csv"))
                scale_values['variable'] = model_dict[model_name]['input_variables']
                scale_values = scale_values.set_index('variable')

                print('Loading storm patches...')
                input_data, output, meta = load_patch_files(config["run_start_date"],
                                                            config["run_end_date"],
                                                            config["run_freq"],
                                                            config["data_path"],
                                                            model_dict[model_name]["input_variables"],
                                                            model_dict[model_name]["output_variables"],
                                                            config["patch_meta_variables"],
                                                            model_dict[model_name]["patch_radius"])

                input_combined = combine_patch_data(input_data, model_dict[model_name]["input_variables"])
                input_scaled, scale_values = min_max_scale(input_combined, scale_values)
                meta_df = get_meta_scalars(meta)
                models[model_name] = load_conv_net(model_path, model_name)
                print(model_name, f'({model_type})')
                print(models[model_name].model_.summary())

                if model_type == 'semi_supervised':

                    neuron_columns = [f"neuron_{n:03d}" for n in range(models[model_name].dense_neurons)]
                    neuron_activations[model_name] = pd.merge(meta_df, pd.DataFrame(0, columns=neuron_columns,
                                                              index=meta_df.index), left_index=True, right_index=True)
                    neuron_activations[model_name].loc[:, neuron_columns] = \
                        models[model_name].output_hidden_layer(input_scaled.values)

                    gmms[model_name] = joblib.load(join(f"{model_path}_GMM_1.mod"))
                    cluster_assignments = joblib.load(join(model_path, f'{model_name}_GMM_1_gmm_labels.dict'))

                    labels[model_name] = predict_labels_gmm(neuron_activations[model_name], gmms[model_name],
                                                            model_name, cluster_assignments)

                elif model_type == 'supervised':

                    labels[model_name] = predict_labels_cnn(input_scaled, models[model_name], model_name)

            elif model_type == 'supervised_DNN':
                models[model_name] = load_model(join(model_dict[model_name]['model_path'], f"{model_name}.h5"),
                                                custom_objects={"brier_score": brier_score, "brier_skill_score": bss})
                print(model_name, f'({model_type})')
                print(models[model_name].summary())
                with open(join(model_dict[model_name]["model_path"], 'scale_values.yaml'), "r") as config_file:
                    scale_values = yaml.load(config_file, Loader=yaml.Loader)
                labels[model_name] = predict_labels_dnn(storm_data, scale_values, models[model_name],
                                                        model_dict[model_name]["input_variables"],
                                                        model_name)

    all_labels = merge_labels(labels, storm_data, config["csv_meta_variables"], config["storm_variables"])
    save_labels(all_labels, config['run_start_date'], config['run_end_date'],
                config['run_freq'], config['output_path'], config['output_format'])


    if args.eval:
        storm_report_path = config["storm_report_path"]
        if not path.exists(storm_report_path):
            makedirs(storm_report_path, exist_ok=False)
        start_date =(pd.Timestamp(config["run_start_date"], tz="UTC")).strftime("%Y%m%d0000")
        end_date = (pd.Timestamp(config["run_end_date"], tz="UTC")).strftime("%Y%m%d0000")
        for report_type in ['filtered_torn', 'filtered_hail', 'filtered_wind']:
            print(f'Downloading SPC storm reports from {start_date} through {end_date} for {report_type}')
            fetch_storm_reports(start_date, end_date, storm_report_path, report_type)

        if not isfile(join(model_dict[model_name]["model_path"], "labels", f"obs_{date_str}_HRRR_hourly.nc")):
            print(f'Aggregating storm reports to a grid.')
            obs = generate_obs_grid(beg=start_date,
                                    end=end_date,
                                    storm_report_path=storm_report_path,
                                    model_grid_path=config["model_grid_path"],
                                    proj_str=config["proj_str"])
            file_name = join(model_dict[model_name]["model_path"], "labels", f"obs_{date_str}_HRRR_hourly.nc")
            obs.to_netcdf(file_name)
            print(f"Wrote {file_name}.")

        print("Aggregating storm mode labels to a grid.")
        print(config["bin_width"])
        data = generate_mode_grid(beg=start_date,
                                  end=end_date,
                                  labels=labels[model_name],
                                  model_grid_path=config["model_grid_path"],
                                  min_lead_time=1,
                                  max_lead_time=24,
                                  proj_str=config["proj_str"],
                                  run_date_freq='1d',
                                  bin_width=config["bin_width"])

        file_name = join(model_dict[model_name]["model_path"], "labels",
                         f"{config['physical_model']}_{model_name}_gridded_labels_{date_str}_HRRR_hourly.nc")
        data.to_netcdf(file_name)
        print(f"Wrote {file_name}.")
    return


if __name__ == "__main__":
    main()
