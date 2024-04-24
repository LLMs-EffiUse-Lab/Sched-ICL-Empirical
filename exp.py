from prediction_model.help import *
from fitness_function.MLP import *
from algs.pygmo_lib import *
from algs.pymoo_lib import *
from algs.spea2 import *
from algs.get_true_pf import *

if __name__ == '__main__':
    itr = 0  # sys.argv[1]

    data_name = f"dataset/logdata.csv"
    system_pair = pd.read_csv(f"dataset/data_selection.csv")

    system_list = ['Android', 'Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux', 'Mac', 'OpenSSH',
                   'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Windows', 'Zookeeper']
    system_list = sorted(system_list, key=lambda x: x.lower())
    model_list = ["standard", "enhance", "simple", "fewshot_1", "fewshot_2", "fewshot_4"]

    test_size = 0.95
    test_sys = 'Android'
    evl_acc_content = []
    evl_acc_query = []

    save_dir = f"res/prediction/{test_sys}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Test System: {test_sys}")

    df_pre_accuracy, df_true_accuracy, df_cost, evl_accuracy_content = individual_classifier_(data_name, model_list,
                                                                                              test_sys, test_size,
                                                                                              itr)

    evl_acc_content.append(evl_accuracy_content)

    spea2_res = spea2(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination=500).run()
    spea2_res.to_csv(f"{save_dir}/spea2_res_{itr}.csv")

    moaco_res = MOACO(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination=500).run()
    moaco_res.to_csv(f"{save_dir}/moaco_res_{itr}.csv")

    rnsga2_res = rnsga2(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination=500).run()
    rnsga2_res.to_csv(f"{save_dir}/rnsga2_res_{itr}.csv")

    nsga2_res = nsga2(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination=500).run()
    nsga2_res.to_csv(f"{save_dir}/nsga2_res_{itr}.csv")

    mopso_res = MOPSO(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination=500).run()
    mopso_res.to_csv(f"{save_dir}/mopso_res_{itr}.csv")

    moead_res = MOEAD(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination=500).run()
    moead_res.to_csv(f"{save_dir}/moead_res_{itr}.csv")

    moeadgen_res = MOEADGEN(df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination=500).run()
    moeadgen_res.to_csv(f"{save_dir}/moeadgen_res_{itr}.csv")

    if itr == '0':
        true_pt, true_pt_solution = get_true_pareto_front(df_true_accuracy, df_cost, model_list).run()
        true_pt.to_csv(f"{save_dir}/true_pt.csv")



