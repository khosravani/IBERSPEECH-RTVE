#!/usr/bin/perl
#
# Apache 2.0.
# Usage: make_sre.pl <path-to-data> <name-of-source> <sre-ref> <output-dir>

if (@ARGV != 4) {
  print STDERR "Usage: $0 <path-to-data> <name-of-source> <sre-ref> <output-dir>\n";
  exit(1);
}

($db_base, $sre_name, $sre_ref_filename, $out_dir) = @ARGV;
%utt2wav = ();

if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir";
}

if (system("find $db_base -name '*.wav' > $out_dir/wav.list") != 0) {
  die "Error getting list of sph files";
}
open(WAVLIST, "<", "$out_dir/wav.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $wav = $_;
  @A1 = split("/",$wav);
  @A2 = split("[./]",$A1[$#A1]);
  $uttId=$A2[0];
  $utt2wav{$uttId} = $wav;
}

open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
open(SRE_REF, "<", $sre_ref_filename) or die "Cannot open SRE reference.";
while (<SRE_REF>) {
  chomp;
  ($other_sre_name, $utt_id, $channel) = split(" ", $_);
  $channel_num = "1";
  if ($channel eq "A") {
    $channel_num = "1";
  } else {
    $channel_num = "2";
  }
  if (($other_sre_name eq $sre_name) and (exists $utt2wav{$utt_id})) {
#   if ((exists $utt2wav{$utt_id})) {
    $full_utt_id = "$other_sre_name-$utt_id-$channel";
    print WAV "$full_utt_id"," sox ",$utt2wav{$utt_id}," -r 8k -b 16 -t wav - remix $channel_num |\n";
  }
}
close(WAV) || die;
close(SRE_REF) || die;
