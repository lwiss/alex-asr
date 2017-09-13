#include "src/decoder_config.h"
#include "libs/kaldi/src/base/kaldi-common.h"
#include "libs/kaldi/src/util/common-utils.h"
#include "libs/kaldi/src/matrix/kaldi-matrix.h"
using namespace kaldi;

namespace alex_asr {

    DecoderConfig::DecoderConfig() :
            lda_mat(NULL),
            spkr_mat(NULL),
            cmvn_mat(NULL),
            ivector_extraction_info(NULL),
            bits_per_sample(16),
            use_lda(false),
            use_ivectors(false),
            use_cmvn(false),
            use_pitch(false),
            cfg_decoder(""),
            cfg_decodable(""),
            cfg_mfcc(""),
            cfg_fbank(""),
            cfg_cmvn(""),
            cfg_splice(""),
            cfg_delta(""),
            cfg_endpoint(""),
            cfg_ivector(""),
            cfg_pitch(""),
            spkrID(""),
            transform_reader(NULL)
    {
        decodable_opts.acoustic_scale = 0.1;
        nnet3_decodable_opts.acoustic_scale = 0.1;
        splice_opts.left_context = 0;
        splice_opts.right_context = 0;
    }

    DecoderConfig::~DecoderConfig() {
        delete lda_mat;
        lda_mat = NULL;
        delete spkr_mat;
        spkr_mat = NULL;
        delete cmvn_mat;
        cmvn_mat = NULL;
        delete ivector_extraction_info;
        ivector_extraction_info = NULL;
        delete transform_reader;
        transform_reader = NULL;
    }

    void DecoderConfig::Register(ParseOptions *po) {
        po->Register("model_type", &model_type_str, "Type of model. GMM/NNET2/NNET3");
        po->Register("feature_type", &feature_type_str, "Type of features. MFCC/FBANK");
        po->Register("model", &model_rxfilename, "Accoustic model filename.");
        po->Register("hclg", &fst_rxfilename, "HCLG FST filename.");
        po->Register("words", &words_rxfilename, "Word to ID mapping filename.");
        po->Register("word_boundary", &word_boundary_rxfilename, "data/lang/phones/word_boundary.int");
        po->Register("mat_lda", &lda_mat_rspecifier, "LDA matrix filename.");
        po->Register("trans_file", &transform_rspecifier, "transform filename.");
        po->Register("mat_cmvn", &fcmvn_mat_rspecifier, "CMVN matrix filename.");
        po->Register("use_lda", &use_lda, "Are we using LDA transform?");
        po->Register("spkrID", &spkrID, "Speaker ID in configuration file");
        po->Register("use_ivectors", &use_ivectors, "Are we using ivector features?");
        po->Register("use_cmvn", &use_cmvn, "Are we using cmvn transform?");
        po->Register("use_pitch", &use_pitch, "Are we using pitch feature?");
        po->Register("bits_per_sample", &bits_per_sample, "Bits per sample for input.");

        po->Register("cfg_decoder", &cfg_decoder, "");
        po->Register("cfg_decodable", &cfg_decodable, "");
        po->Register("cfg_mfcc", &cfg_mfcc, "");
        po->Register("cfg_fbank", &cfg_fbank, "");
        po->Register("cfg_cmvn", &cfg_cmvn, "");
        po->Register("cfg_splice", &cfg_splice, "");
        po->Register("cfg_delta", &cfg_delta, "");
        po->Register("cfg_endpoint", &cfg_endpoint, "");
        po->Register("cfg_ivector", &cfg_ivector, "");
        po->Register("cfg_pitch", &cfg_pitch, "");
    }

    void DecoderConfig::LoadConfigs(const string cfg_file) {
        std::string model_path("");

        ParseOptions po("");
        Register(&po);

        KALDI_VLOG(2) << "Reading master config file: " << cfg_file;
        po.ReadConfigFile(cfg_file);

        if(model_type_str == "nnet3") {
            LoadConfig(cfg_decodable, &nnet3_decodable_opts);
        } else {
            LoadConfig(cfg_decodable, &decodable_opts);
        }

        LoadConfig(cfg_decoder, &decoder_opts);
        LoadConfig(cfg_mfcc, &mfcc_opts);
        LoadConfig(cfg_fbank, &fbank_opts);
        LoadConfig(cfg_cmvn, &cmvn_opts);
        LoadConfig(cfg_splice, &splice_opts);
        LoadConfig(cfg_delta, &delta_opts);
        LoadConfig(cfg_endpoint, &endpoint_config);
        LoadConfig(cfg_ivector, &ivector_config);
        LoadConfig(cfg_pitch, &pitch_opts);
        LoadConfig(cfg_pitch, &pitch_process_opts);

        InitAux();
    }

    void DecoderConfig::InitAux() {
        if(use_lda) {
            LoadLDA();
        }

        if (transform_rspecifier != "") {
            if (transform_rspecifier.substr(0,4)!= "ark:"){
                std::string fullpath = std::string(realpath(transform_rspecifier.c_str(), NULL));
                transform_rspecifier = "ark:" + fullpath;
            }
            KALDI_PARANOID_ASSERT(transform_reader == NULL);
            transform_reader = new RandomAccessBaseFloatMatrixReader(transform_rspecifier);
        }

        if (spkrID!="" && spkrID!="None" && spkrID!="NoSpkrID"){
            bool res = true; 
            res &= OptionCheck(transform_rspecifier == "",
                           "You have to specify --trans_file when you specify --spkrID.");
            LoadSpkrTransform();
        }

        if (use_cmvn) {
            LoadCMVN();
        }

        if (use_ivectors) {
            LoadIvector();
        }
    }

    void DecoderConfig::LoadLDA() {
        KALDI_VLOG(2) << "Loading LDA matrix.";
        bool binary_in;
        Input ki(lda_mat_rspecifier, &binary_in);

        KALDI_PARANOID_ASSERT(lda_mat == NULL);
        lda_mat = new Matrix<BaseFloat>();
        lda_mat->Read(ki.Stream(), binary_in);
    }

    void DecoderConfig::LoadSpkrTransform(){
        KALDI_VLOG(2) << "Loading transform file";
        KALDI_VLOG(2) << "the Speaker ID is " << spkrID;
        KALDI_PARANOID_ASSERT(transform_reader != NULL);
        const Matrix<BaseFloat> &trans = transform_reader->Value(spkrID);

        KALDI_PARANOID_ASSERT(spkr_mat == NULL);
        spkr_mat = new Matrix<BaseFloat>(trans);
    }

    void DecoderConfig::LoadCMVN() {
        KALDI_VLOG(2) << "Loading global CMVN stats.";
        bool binary_in;
        Input ki(fcmvn_mat_rspecifier, &binary_in);

        KALDI_PARANOID_ASSERT(cmvn_mat == NULL);
        cmvn_mat = new Matrix<double>();
        cmvn_mat->Read(ki.Stream(), binary_in);
    }

    void DecoderConfig::LoadIvector() {
        KALDI_LOG << "Loading IVector extraction info.";
        ivector_extraction_info = new OnlineIvectorExtractionInfo(ivector_config);
    }

    template<typename C>
    void DecoderConfig::LoadConfig(string file_name, C *opts) {
        if (FileExists(file_name)) {
            ReadConfigFromFile(file_name, opts);
            KALDI_VLOG(2) << "Config loaded: " << file_name;
        } else {
            KALDI_VLOG(2) << "Config not found: " << file_name;
        }
    }

    bool DecoderConfig::FileExists(string strFilename) {
        struct stat stFileInfo;
        bool blnReturn;
        int intStat;

        // Attempt to get the file attributes
        intStat = stat(strFilename.c_str(), &stFileInfo);
        if (intStat == 0) {
            // We were able to get the file attributes
            // so the file obviously exists.
            blnReturn = true;
        } else {
            // We were not able to get the file attributes.
            // This may mean that we don't have permission to
            // access the folder which contains this file. If you
            // need to do that level of checking, lookup the
            // return values of stat which will give you
            // more details on why stat failed.
            blnReturn = false;
        }

        return blnReturn;
    }

    bool DecoderConfig::InitAndCheck() {
        bool res = true;

        if(model_type_str == "gmm") {
            model_type = GMM;
        } else if(model_type_str == "nnet2") {
            model_type = NNET2;
        } else if(model_type_str == "nnet3") {
            model_type = NNET3;
        } else {
            res = false;

            KALDI_ERR << "You have to specify a valid model_type.";
        }

        if(feature_type_str == "mfcc" || feature_type_str == "") {
            feature_type = MFCC;
        } else if(feature_type_str == "fbank") {
            feature_type = FBANK;
        } else {
            res = false;

            KALDI_ERR << "You have to specify a valid feature_type.";
        }

        res &= OptionCheck(use_ivectors && cfg_ivector == "",
                           "You have to specify --cfg_ivector if you want to use ivectors.");
        res &= OptionCheck(use_cmvn && fcmvn_mat_rspecifier == "",
                           "You have to specify --cfg_cmvn if you want to use CMVN.");
        res &= OptionCheck(use_pitch && cfg_pitch == "",
                           "You have to specify --cfg_pitch if you want to use pitch.");

        res &= OptionCheck(model_rxfilename == "",
                           "You have to specify --model.");

        res &= OptionCheck(fst_rxfilename == "",
                           "You have to specify --hclg.");

        res &= OptionCheck(words_rxfilename == "",
                           "You have to specify --words.");

        res &= OptionCheck(use_lda && lda_mat_rspecifier == "",
                           "You have to specify --mat_lda or set --use_lda=false.");

        return res;
    }

    bool DecoderConfig::OptionCheck(bool cond, std::string fail_text) {
        if (cond) {
            KALDI_ERR << fail_text;
            return false;
        }
        return true;
    }

    BaseFloat DecoderConfig::FrameShiftInSeconds() const {
        int frame_subsampling_factor = 1;
        if(model_type == DecoderConfig::NNET3) {
            frame_subsampling_factor = nnet3_decodable_opts.frame_subsampling_factor;
        }

        if(feature_type == DecoderConfig::MFCC) {
            return mfcc_opts.frame_opts.frame_shift_ms * frame_subsampling_factor * 1.0e-03;
        } else if(feature_type == DecoderConfig::FBANK) {
            return fbank_opts.frame_opts.frame_shift_ms * frame_subsampling_factor * 1.0e-03;
        } else {
            KALDI_ERR << "You have to specify a valid feature_type.";
            return 0.0;
        }
    }

    BaseFloat DecoderConfig::SamplingFrequency() const {
        if(feature_type == DecoderConfig::MFCC) {
            return mfcc_opts.frame_opts.samp_freq;
        } else if(feature_type == DecoderConfig::FBANK) {
            return fbank_opts.frame_opts.samp_freq;
        } else {
            KALDI_ERR << "You have to specify a valid feature_type.";
            return 0.0;
        }
    }

    void DecoderConfig::ChangeSpkrID(string spkr_ID){
        this->spkrID = spkr_ID;
        if(spkr_ID!="" && spkr_ID!="None" && spkr_ID!="NoSpkrID"){
            delete(spkr_mat);
            spkr_mat = NULL;
            bool res = true; 
            res &= OptionCheck(transform_rspecifier == "",
                           "You have to specify --trans_file when you specify --spkrID.");
            LoadSpkrTransform();    
        }
    }

    vector<string> DecoderConfig::GetIDList(){
        std::vector<string> speakers;
        // KALDI_PARANOID_ASSERT(speaker_reader == NULL);
        SequentialBaseFloatMatrixReader speaker_reader(transform_rspecifier);

        if (transform_rspecifier != ""){
            int num_done = 0;
            while (!speaker_reader.Done()) {
                string key = speaker_reader.Key();
                speaker_reader.Next(); 
                num_done++; 
                speakers.push_back(key);
            }
        }
        
        return speakers;
    }
}
