use coupled_chan_cli::CoupledChanCli;
use raf_ra::available_problems;

fn main() -> anyhow::Result<()> {
    CoupledChanCli::parse().execute(crate::available_problems())
}
