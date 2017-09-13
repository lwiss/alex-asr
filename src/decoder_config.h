#ifndef ALEX_ASR_DECODER_CONFIG_H_
#define ALEX_ASR_DECODER_CONFIG_H_

#include <sys/stat.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "decoder/lattice-faster-decoder.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "feat/feature-mfcc.h"
#include "feat/online-feature.h"
#include "feat/pitch-functions.h"
#include "nnet2/online-nnet2-decodable.h"
#include "nnet3/online-nnet3-decodable-simple.h"
#include "online2/online-endpoint.h"
#include "online2/online-ivector-feature.h"
#include "util/stl-utils.h"
#include "src/utils.h"


using namespace kaldi;

namespace alex_asr {
    class DecoderConfig {
    public:
        enum ModelType { NoneModelType, GMM, NNET2, NNET3 };
        enum FeatureType { NoneFeatureType, MFCC, FBANK };

        //DecoderConfig();
        DecoderConfig();
        ~DecoderConfig();
        void Register(ParseOptions *po);
        void LoadConfigs(const string cfg_file);
        void ChangeSpkrID(string spkr_ID);
        bool InitAndCheck();
        BaseFloat FrameShiftInSeconds() const;
        BaseFloat SamplingFrequency() const;
        vector<string> GetIDList();

        LatticeFasterDecoderConfig decoder_opts;
        nnet2::DecodableNnet2OnlineOptions decodable_opts;
        nnet3::DecodableNnet3OnlineOptions nnet3_decodable_opts;
        MfccOptions mfcc_opts;
        FbankOptions fbank_opts;
        OnlineCmvnOptions cmvn_opts;
        OnlineSpliceOptions splice_opts;
        DeltaFeaturesOptions delta_opts;
        OnlineEndpointConfig endpoint_config;
        OnlineIvectorExtractionConfig ivector_config;
        PitchExtractionOptions pitch_opts;
        ProcessPitchOptions pitch_process_opts;

        Matrix<BaseFloat> *lda_mat;
        Matrix<BaseFloat> *spkr_mat;
        Matrix<double> *cmvn_mat;
        OnlineIvectorExtractionInfo *ivector_extraction_info;

        ModelType model_type;
        FeatureType feature_type;
        int32 bits_per_sample;

        bool use_lda;
        bool use_delta;
        bool use_ivectors;
        bool use_cmvn;
        bool use_pitch;

        std::string cfg_decoder;
        std::string cfg_decodable;
        std::string cfg_mfcc;
        std::string cfg_fbank;
        std::string cfg_cmvn;
        std::string cfg_splice;
        std::string cfg_delta;
        std::string cfg_endpoint;
        std::string cfg_ivector;
        std::string cfg_pitch;

        std::string model_rxfilename;
        std::string fst_rxfilename;
        std::string words_rxfilename;
        std::string word_boundary_rxfilename;
        std::string lda_mat_rspecifier;
        std::string transform_rspecifier;
        std::string fcmvn_mat_rspecifier;
        std::string spkrID;
    private:
        void InitAux();
        void LoadSpkrTransform();
        void LoadLDA();
        void LoadCMVN();
        void LoadIvector();
        template<typename C> void LoadConfig(string file_name, C *opts);
        bool FileExists(string strFilename);
        bool OptionCheck(bool cond, std::string fail_text);
        RandomAccessBaseFloatMatrixReader *transform_reader;
        // SequentialBaseFloatMatrixReader *speaker_reader;

        string model_type_str;
        string feature_type_str;
    };
}

#endif //PYKALDI_PYKALDI2_DECODER_CONFIG_H
