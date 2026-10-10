#![forbid(unsafe_code)]

fn main() {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments == ["--version"] {
        println!(concat!(
            env!("CARGO_BIN_NAME"),
            " ",
            env!("CARGO_PKG_VERSION")
        ));
        return;
    }
    let command = match synapse_certify_runner::command::parse(&arguments) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let source = synapse_certify::command::source_stamp(
        option_env!("SYNAPSE_BUILD_REV"),
        option_env!("SYNAPSE_BUILD_TREE"),
    );
    match synapse_certify_runner::command::dispatch(command, source) {
        Ok(record) => println!("{record}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
