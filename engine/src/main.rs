use clap::Parser;
use renderer::Core;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_parser)]
    width: Option<u32>,

    #[clap(short, long, value_parser)]
    height: Option<u32>,

    #[clap(short, long)]
    igpu: bool,
}

struct LaunchOptions {
    width: u32,
    height: u32,
    integrated: bool,
}

impl From<&Cli> for LaunchOptions {
    fn from(cli: &Cli) -> Self {
        LaunchOptions {
            width: cli.width.unwrap_or(1280),
            height: cli.height.unwrap_or(720),
            integrated: cli.igpu,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let options = LaunchOptions::from(&cli);

    let renderer = Core::new(options.width, options.height, options.integrated);
}


