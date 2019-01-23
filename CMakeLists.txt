#include "Predictor.h"
#include "Utils/Systems.h"
#include "Utils/Timer.h"
#include <cassert>
#include <algorithm>

void Predictor::init() {
	assert(!_cpp_files.empty() && "Empty files to be processed!");
	assert(opts.log_dir != "" && "Empty log dir!");
	if (!opts.generate_graph_only) {
		assert(opts.buggy_dot_file != "" && "Empty buggy dot file!");
		assert(opts.fine_dot_file != "" && "Empty fine dot file!");
		_buggy_graph = read_graphviz_graph(opts.buggy_dot_file.c_str());
		std::vector<std::string> fine_graph_files = _sys.split_list(opts.fine_dot_file);
		for(std::string fine_graph_file : fine_graph_files)
			_fine_graph.push_back(read_graphviz_graph(fine_graph_file.c_str()));
	}
	_logger.set_rank(opts.log_rank);
}

string Predictor::parse_source_name(string input) {
	Systems sys;
	string file_name = sys.get_path_short_name(input);
	string::size_type iPos = file_name.find_last_of('#');
	if (iPos != -1) {
		file_name = file_name.substr(0, iPos);
		file_name += ".cpp";
	}
	else {
		iPos = file_name.find_last_of('.');
		file_name = file_name.substr(0, iPos);
		file_name += ".cpp";
	}

	return file_name;

}

string Predictor::rename_file(string input) {
	Systems sys;
	string file_name = sys.get_path_short_name(input);
	string::size_type iPos = file_name.find_last_of('.');
	if (iPos != -1) {
		file_name = file_name.substr(0, iPos);
		file_name += "#inline.cpp";
	}

	return file_name;
}

bool Predictor::get_graph_output_dir(string input, string code_name,
	string &output_dir) {
	string dir = _sys.get_file_dir_path(input);
	_sys.string_replace(dir, opts.input_dir, opts.work_dir);
	string::size_type iPos = code_name.find_last_of('.');
	string dir_name = code_name.substr(0, iPos);
	output_dir = dir + "\\" + dir_name;
	if (_sys.is_dir(output_dir)) {
		set<string> suffix;
		suffix.insert("dot");
		vector<string> files;
		vector<string> file_names;
		_sys.get_files(output_dir, files, file_names, suffix);
		//  未生成dot
		if (files.empty())
			return false;
		return true;
	}
	if (_sys.create_folder(output_dir))
		return false;
}

bool Predictor::get_result_output_dir(string input, string code_name,
	string &result_dir) {
	string dir = _sys.get_file_dir_path(input);
	_sys.string_replace(dir, opts.input_dir, opts.result_dir);
	string::size_type iPos = code_name.find_last_of('.');
	string dir_name = code_name.substr(0, iPos);
	result_dir = dir + "\\" + dir_name;
	if (_sys.is_dir(result_dir)) {
		return true;
	}
	if (_sys.create_folder(result_dir))
		return false;
}

// stupid
string Predictor::get_clang_command(string cpp_file, string bc_file,
	string output_dir) {
	Systems sys;
	string clang_cl_command = "clang-cl -### -Xclang -emit-llvm /c /Z7";
	if (opts.clang_cl_args != "") {
		vector<string> clang_cl_arg_list = sys.split_list(opts.clang_cl_args);
		for (string dg_arg : clang_cl_arg_list) {
			clang_cl_command += " ";
			clang_cl_command += dg_arg;
		}
	}
	clang_cl_command += " -Xclang -fcxx-exceptions -fms-compatibility-version=19 -std=c++11 " + cpp_file +
		" -o " + bc_file + " 2>" + output_dir + "\\bc-command.log";
	if (opts.exe_dir != "")
		clang_cl_command = opts.exe_dir + "\\" + clang_cl_command;
	string clang_cl_result = sys.exec(clang_cl_command);
	if (clang_cl_result == "error")
		return "error";
	clang_cl_result = sys.read_file_to_a_string(output_dir + "\\bc-command.log");
	size_t pos = clang_cl_result.find("clang-cl.exe\" ");
	if (pos == string::npos)
		return "error";
	return clang_cl_result.substr(pos + 14);
}

bool Predictor::generate_graph(string target_file,
	string code_name, 
	string graph_output_dir,
	Logger *out_logger) {
	Timer timer;
	vector<string> pass_options;
	if (opts.pass_options != "")
		pass_options = _sys.split_list(opts.pass_options);
	string new_filepath = graph_output_dir + "\\" + code_name;
	string bc_filepath = new_filepath + ".bc";
	timer.start();
	CopyFile(target_file.c_str(), new_filepath.c_str(), TRUE);

	out_logger->info_out("Generating bc file...\n");
#ifndef NDEBUG
	_logger.info_screen("Generating bc file...\n");
#endif // !NDEBUG
	string generate_bc_file;
	string bc_result;
	generate_bc_file = get_clang_command(new_filepath, bc_filepath,
		graph_output_dir);
	if (generate_bc_file == "error") {
		out_logger->error_out("Failed generate bc command for %s\n", target_file.c_str());
#ifndef NDEBUG
		_logger.error_screen("Failed generate bc command for %s\n", target_file.c_str());
#endif // !NDEBUG
		return false;
	}
	generate_bc_file = "clang " + generate_bc_file;
	generate_bc_file = generate_bc_file + " 2>" + graph_output_dir + "\\bc.log";
	bc_result = _sys.exec(generate_bc_file);
	if (bc_result == "error") {
		out_logger->error_out("Failed generate bc file for %s\n", target_file.c_str());
#ifndef NDEBUG
		_logger.error_screen("Failed generate bc file for %s\n", target_file.c_str());
#endif // !NDEBUG
		return false;
	}

	out_logger->info_out("Dumping graph files...\n");
#ifndef NDEBUG
	_logger.info_screen("Dumping graph files...\n");
#endif // !NDEBUG
	// llvmdg::LLVMDependenceGraphOptions dgOptions;
	// dgOptions.maxCallDepth = opts.max_call_depth;
	// dgOptions.PTAOptions.maxCallDepth = opts.max_call_depth;
	// dgOptions.RDAOptions.maxCallDepth = opts.max_call_depth;
	// debug::DGDumpOptions dgDumpOptions;
	// dgDumpOptions.dropBB = true;
	DumperOptions dumpOptions;
	dumpOptions.bcFile = bc_filepath;
	// dumpOptions.outputDir = graph_output_dir;
	dumpOptions.keepFiles = std::set<std::string>{ code_name };
	// dumpOptions.dgOptions = dgOptions;
	// dumpOptions.dgDumpOptions = dgDumpOptions;

	Dumper dumper(dumpOptions, out_logger);
	std::vector<llvm::Function *> funcs = dumper.getRelatedFunctions();
	// Maybe failed parse the module.
	if (funcs.empty()) {
		out_logger->warning_out("No related functions for %s.\n", target_file.c_str());
#ifndef NDEBUG
		_logger.warning_screen("No related functions for %s.\n", target_file.c_str());
#endif // !NDEBUG
		return true;
	}
	bool should_update = opts.donot_use_old;
	if (!should_update) {
		std::set<std::string> suffix = { "dot" };
		std::vector<std::string> exist_dot_files;
		std::vector<std::string> exist_dot_filenames;
		_sys.get_files(graph_output_dir, exist_dot_files, exist_dot_filenames, suffix, false);
		if (funcs.size() > exist_dot_files.size())
			should_update = true;
	}

	if (should_update) {
		string dg_dump_dot = "llvm-dg-dump -drop-bb ";
		if (opts.dg_args != "") {
			vector<string> dg_arg_list = _sys.split_list(opts.dg_args);
			for (string dg_arg : dg_arg_list) {
				dg_dump_dot += dg_arg;
				if (dg_arg == "-call-depth")
					dg_dump_dot += "=";
				else
					dg_dump_dot += " ";
			}
		}
		dg_dump_dot = dg_dump_dot + "-input " + bc_filepath +
			" -outdir " + graph_output_dir +
			" -keepfiles " + code_name;
		if (!opts.dg_log)
			dg_dump_dot = dg_dump_dot + " 2>nul";
		string dg_dump_dot_result = _sys.exec(dg_dump_dot);
		if ("error" == dg_dump_dot_result) {
			out_logger->error_out("Failed generate graph for \n\t%s.---\n", target_file.c_str());
#ifndef NDEBUG
			_logger.error_screen("Failed generate graph for \n\t%s.---\n", target_file.c_str());
#endif // !NDEBUG
			return false;
		} else if(_strings.endswith(dg_dump_dot_result, "llvm-dg-dump error!")) {
			out_logger->error_out("Failed generate graph for \n\t%s\n\terror: %s.---\n", 
				target_file.c_str(), "llvm-dg-dump error!");
#ifndef NDEBUG
			_logger.error_screen("Failed generate graph for \n\t%s\n\terror: %s.---\n", 
				target_file.c_str(), "llvm-dg-dump error!");
#endif // !NDEBUG
			return false;
		}
	} else {
		out_logger->info_out("Use the old result files for %s.\n", target_file.c_str());
#ifndef NDEBUG
		_logger.info_screen("Use the old result files for %s.\n", target_file.c_str());
#endif // !NDEBUG
	}
	// 删除临时cpp文件
	_sys.delete_file(new_filepath);
	// 删除bc，文件太大，占用空间
	if (!opts.donot_delete_bc)
		_sys.delete_file(bc_filepath);
	
	timer.stop();
	out_logger->info_out("%s Generate graph files for %s took %s.\n",
		timer.get_time().c_str(), target_file.c_str(), timer.get_second_elapsed().c_str());
#ifndef NDEBUG
	_logger.info_screen("%s Generate graph files for %s took %s.\n",
		timer.get_time().c_str(), target_file.c_str(), timer.get_second_elapsed().c_str());
#endif // !NDEBUG
	return true;
}

static string get_line_info(string label_url) {
	if (label_url == "")
		return "";
	utils::Systems sys;
	string result;
	vector<string> tags = sys.split_list(label_url, ':');
	size_t length = tags.size();
	if (length <= 2)
		return "";
	result += tags[length - 2];
	result += ":";
	result += tags[length - 1];
	return result;
}

void Predictor::predict_result(vector<IndexContainT<int, PairContainerT<string, string>>> bug_result,
	vector<IndexContainT<int, PairContainerT<string, string>>> fine_result,
	pugi::xml_node &func_node) {
	vector<set<string>> bug_result_lines;
	vector<set<string>> fine_result_lines;
	for (IndexContainT<int, PairContainerT<string, string>> pair : bug_result) {
		set<string> lines;
		for (int i = 0; i < pair.size(); i++) {
			PairContainerT<string, string> p = pair.get_index(i);
			string line_info = get_line_info(p.get_value());
			if (line_info != "")
				lines.insert(line_info);
		}
		bug_result_lines.push_back(lines);
	}
	for (IndexContainT<int, PairContainerT<string, string>> pair : fine_result) {
		set<string> lines;
		for (int i = 0; i < pair.size(); i++) {
			PairContainerT<string, string> p = pair.get_index(i);
			string line_info = get_line_info(p.get_value());
			if (line_info != "")
				lines.insert(line_info);
		}
		fine_result_lines.push_back(lines);
	}
	for (int i = 0; i < bug_result_lines.size(); i++) {
		set<string> bug_lines = bug_result_lines[i];
		bool buggy = true;
		int j = 0;
		for (; j < fine_result_lines.size(); j++) {
			set<string> fine_lines = fine_result_lines[j];
			vector<string> intersection;
			set_intersection(bug_lines.begin(), bug_lines.end(), 
				fine_lines.begin(), fine_lines.end(), 
				inserter(intersection, intersection.begin()));
			// confirmed fine
			// if (intersection.size() == bug_lines.size()) {
			if (intersection.size() > 0) {
				buggy = false;
				break;
			}
		}
		if (buggy) {
			pugi::xml_node buggy_node = func_node.append_child("bug");
			for (int k = 0; k < bug_result[i].size(); k++) {
				PairContainerT<string, string> p = bug_result[i].get_index(k);
				pugi::xml_node pair_node = buggy_node.append_child("pair");
				pair_node.append_attribute("buggy").set_value(p.get_key().c_str());
				pair_node.append_attribute("new").set_value(p.get_value().c_str());
			}
		}
		else {
			pugi::xml_node buggy_node = func_node.append_child("fine");
			for (int k = 0; k < fine_result[j].size(); k++) {
				PairContainerT<string, string> p = fine_result[j].get_index(k);
				pugi::xml_node pair_node = buggy_node.append_child("pair");
				pair_node.append_attribute("fine").set_value(p.get_key().c_str());
				pair_node.append_attribute("new").set_value(p.get_value().c_str());
			}
		}
	}
	return;
}

bool Predictor::execute_iso(string work_dir, 
	pugi::xml_node &code_node,
	Logger *out_logger,
	Logger *total_logger) {
	Systems sys;
	set<string> suffix;
	suffix.insert("dot");
	vector<string> dot_files;
	vector<string> dot_file_names;
	sys.get_files(work_dir, dot_files, dot_file_names, suffix);
	code_node.append_attribute("funcs").set_value(dot_files.size());
	if (dot_files.empty()) {
		out_logger->warning_out("Empty dot files for %s\n", work_dir.c_str());
#ifndef NDEBUG
		_logger.warning_screen("Empty dot files for %s\n", work_dir.c_str());
#endif // !NDEBUG
		return true;
	}
	out_logger->info_out("Dot files number: %d\n", dot_files.size());
#ifndef NDEBUG
	_logger.info_screen("Dot files number: %d\n", dot_files.size());
#endif // !NDEBUG
	
	Timer timer;
	Timer timer1;
	timer1.start();
	for (string dot_file : dot_files) {
		out_logger->out_file("---FUNCTION: %s\n", sys.get_path_short_name(dot_file).c_str());
		timer.start();
		graph_t target_graph;
		try {
			target_graph = read_graphviz_graph(dot_file.c_str());
		} catch (char* msg) {
			out_logger->error_out("Failed read graph for %s\n\tError: %s", 
				dot_file.c_str(), msg);
			continue;
		} catch (...) {
			out_logger->error_out("Failed read graph for %s\n", dot_file.c_str()); 
			continue;
		}
		timer.stop();
		out_logger->out_file("Read graph took %s\n", timer.get_time_elapsed().c_str());
		out_logger->out_file("Buggy iso: \n");
		vector<IndexContainT<int, PairContainerT<string, string>>> bug_result_label_url;
		timer.start();
		// 图匹配过程
		GraphIso buggy_graph_iso(_buggy_graph, target_graph, out_logger);
		bool bug_result;
		try {
			bug_result = buggy_graph_iso.graph_iso(bug_result_label_url);
		} catch (char* msg) {
			out_logger->error_out("Failed read graph for %s\n\tError: %s",
				dot_file.c_str(), msg);
			continue;
		} catch (...) {
			out_logger->error_out("Failed read graph for %s\n", dot_file.c_str());
			continue;
		}
		timer.stop();
		out_logger->out_file("Iso took %s\n", timer.get_time_elapsed().c_str());
		// 匹配成功
		if (bug_result || !bug_result_label_url.empty()) {
			int map_cnt = 1;
			for (IndexContainT<int, PairContainerT<string, string>> pair : bug_result_label_url) {
				out_logger->out_file("Map %d: ", map_cnt);
				for (int i = 0; i < pair.size(); i++) {
					PairContainerT<string, string> p = pair.get_index(i);
					out_logger->out_file("(%s, %s) ", p.get_key().c_str(), p.get_value().c_str());
				}
				out_logger->out_file("\n");
				map_cnt++;
			}
		}
		else
			out_logger->out_file("No iso.\n");

		out_logger->out_file("Fine iso: \n");
		vector<IndexContainT<int, PairContainerT<string, string>>> fine_result_label_url;
		timer.start();
		// 图匹配过程
		bool fine_result;
		for (graph_t fine_graph : _fine_graph) {
			GraphIso fine_graph_iso(fine_graph, target_graph, out_logger);
			vector<IndexContainT<int, PairContainerT<string, string>>> temp_label_url;;
			try {
				fine_result = fine_graph_iso.graph_iso(temp_label_url);
			}
			catch (...) {
				out_logger->error_out("Failed fine iso for %s\n", dot_file.c_str());
				continue;
			}
			for (IndexContainT<int, PairContainerT<string, string>> temp_map : temp_label_url)
				fine_result_label_url.push_back(temp_map);
		}
		timer.stop();
		out_logger->out_file("Iso took %s\n", timer.get_time_elapsed().c_str());
		// 匹配成功
		if (fine_result || !fine_result_label_url.empty()) {
			int map_cnt = 1;
			for (IndexContainT<int, PairContainerT<string, string>> pair : fine_result_label_url) {
				out_logger->out_file("Map %d: ", map_cnt);
				for (int i = 0; i < pair.size(); i++) {
					PairContainerT<string, string> p = pair.get_index(i);
					out_logger->out_file("(%s, %s) ", p.get_key().c_str(), p.get_value().c_str());
				}
				out_logger->out_file("\n");
				map_cnt++;
			}
		}
		else
			out_logger->out_file("No iso.\n");

		// output result
		pugi::xml_node func_node = code_node.append_child("function");
		func_node.append_attribute("name").set_value(sys.get_path_short_name(dot_file).c_str());
		predict_result(bug_result_label_url, fine_result_label_url, func_node);
	}
	timer1.stop();
	out_logger->info_out("%s Iso took %s.\n\n",
		timer1.get_time().c_str(), timer1.get_second_elapsed().c_str());
#ifndef NDEBUG
	_logger.info_screen("%s Iso took %s.\n", 
		timer1.get_time().c_str(), timer1.get_second_elapsed().c_str());
#endif // !NDEBUG
	return true;
}

bool Predictor::process_file(string file, string code_name,
	Logger *total_logger) {
	bool step_1 = true;
	bool step_2 = true;
	if (file == "")
		return false;
	_timer.start();
	if (code_name == "") {
		code_name = parse_source_name(file);
	}

	std::string result_dir;
	std::string graphs_dir;
	std::string log_file_path;
	get_result_output_dir(file, code_name, result_dir);
	bool exist_graph = get_graph_output_dir(file, code_name, graphs_dir);
	log_file_path = result_dir + "\\" + code_name + ".log";
	Logger out_logger(log_file_path.c_str());
	out_logger.info_out("%s Processing file %s\n", _timer.get_time().c_str(), file.c_str());

	if (!opts.execute_iso_only) {
		// 生成bc文件和每一个函数的graph文件
		out_logger.info_out("---Step1. Generate bitcode file and dump the graph.---\n");
#ifndef NDEBUG
		_logger.info_screen("---Step1. Generate bitcode file and dump the graph.---\n");
#endif // !NDEBUG
		step_1 = generate_graph(file, code_name, graphs_dir, &out_logger);
		out_logger.info_out("---Step1. %s.---\n", step_1 ? "Done" : "Failed");
#ifndef NDEBUG
		_logger.info_screen("---Step1. %s.---\n", step_1 ? "Done" : "Failed");
#endif // !NDEBUG
	}
	// 只生成graph文件
	if (opts.generate_graph_only)
		return true;

	// 图同构
	out_logger.info_out("---Step2. Execute subgraph iso.---\n");
#ifndef NDEBUG
	_logger.info_screen("---Step2. Execute subgraph iso.---\n");
#endif // !NDEBUG
	// output log
	pugi::xml_document log_doc;
	pugi::xml_node pre = log_doc.prepend_child(pugi::node_declaration);
	pre.append_attribute("version") = "1.0";
	pre.append_attribute("encoding") = "UTF-8";
	pre.append_attribute("standalone") = "no";

	pugi::xml_node code_node = log_doc.append_child("code");
	code_node.append_attribute("name").set_value(code_name.c_str());

	step_2 = execute_iso(graphs_dir, code_node, &out_logger, total_logger);
	out_logger.info_out("---Step2. %s.---\n", step_2 ? "Done" : "Failed");
#ifndef NDEBUG
	_logger.info_screen("---Step2. %s.---\n", step_2 ? "Done" : "Failed");
#endif // !NDEBUG

	string out_xml_file = result_dir + "//" + code_name + ".xml";
	log_doc.save_file(out_xml_file.c_str());

	_timer.stop();
	out_logger.info_out("%s Processed file %s took %s\n",
		_timer.get_time().c_str(), file.c_str(), _timer.get_second_elapsed().c_str());
	out_logger.save();
#ifndef NDEBUG
	_logger.max_level_screen("%s Processed file %s took %s\n",
		_timer.get_time().c_str(), file.c_str(), _timer.get_second_elapsed().c_str());
#endif // !NDEBUG

	return true;
}
