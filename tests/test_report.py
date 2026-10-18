from nslkdd.report import plot_report


def test_json_confusion_matrix_can_be_rendered(tmp_path):
    report={'selected_model':'Example','validation_results':[{'model':'Example','f1':.8}],
            'official_test':{'confusion_matrix':[[9,1],[2,8]]}}
    plot_report(report,tmp_path)
    assert (tmp_path/'validation-comparison.png').stat().st_size>1000
    assert (tmp_path/'test-confusion.png').stat().st_size>1000
