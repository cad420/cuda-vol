#pragma once

#include <fstream>
#include <string>
#include <memory>

#include <cudafx/misc.hpp>
#include <cudafx/memory.hpp>
#include <VMUtils/concepts.hpp>
#include <VMUtils/modules.hpp>
#include <vocomp/index/index.hpp>
#include <vocomp/refine/extractor.hpp>
#include <vocomp/video/decompressor.hpp>

VM_BEGIN_MODULE( vol )

using namespace std;
namespace cuda = cufx;

template <typename Voxel>
struct VolumeInner : vm::NoCopy, vm::NoMove
{
	VolumeInner( ifstream &&is, std::size_t is_len,
				 std::string const &method ) :
	  is( std::move( is ) ),
	  reader( this->is, 0, is_len ),
	  _( reader ),
	  block_dim( cuda::Extent{}
				   .set_width( _.block_size() )
				   .set_height( _.block_size() )
				   .set_depth( _.block_size() ) ),
	  block_size( sizeof( Voxel ) * block_dim.size() )
	{
		if ( method == "h264" || method == "hevc" ) {
			auto reader = _.extract(
			  index::Idx{}
				.set_x( 0 )
				.set_y( 0 )
				.set_z( 0 )
			  // _.index().begin()->first
			);  // examine the first block to make sure encoding
			decomp = std::make_shared<vol::video::Decompressor>( reader );
		} else if ( method == "none" ) {
			decomp = std::make_shared<vol::Copy>();
		} else {
			throw std::logic_error( vm::fmt( "unrecognized decompression method: {}", method ) );
		}
	}

	ifstream is;
	StreamReader reader;
	vol::refine::Extractor _;
	cuda::Extent block_dim;
	size_t block_size;
	std::shared_ptr<vol::Pipe> decomp;
};

VM_EXPORT
{
	template <typename Voxel>
	struct Volume;

	template <typename Voxel>
	struct ArchievedVolumeBlock;

	template <typename Voxel>
	struct VolumeBlock
	{
	private:
		struct Inner : vm::NoCopy, vm::NoMove
		{
			string _;
			cuda::Extent dim;
		};

	public:
		const Voxel *data() const
		{
			return reinterpret_cast<const Voxel *>( _->_.c_str() );
		}
		cuda::MemoryView3D<Voxel> view() const
		{
			auto view_info = cuda::MemoryView2DInfo{}
							   .set_stride( this->_->dim.width * sizeof( Voxel ) )
							   .set_width( this->_->dim.width )
							   .set_height( this->_->dim.height );
			return cuda::MemoryView3D<Voxel>( const_cast<char *>( _->_.c_str() ),
											  view_info, this->_->dim );
		}

	private:
		VolumeBlock( string &&_, cuda::Extent dim )
		{
			this->_->_ = std::move( _ );
			this->_->dim = dim;
		}

		shared_ptr<Inner> _ = make_shared<Inner>();
		friend struct ArchievedVolumeBlock<Voxel>;
	};

	template <typename Voxel>
	struct ArchievedVolumeBlock
	{
		VolumeBlock<Voxel> unarchieve() const
		{
			string buffer;
			buffer.resize( _->block_size );
			auto buffer_ptr = const_cast<char *>( buffer.c_str() );
			SliceWriter writer( buffer_ptr, _->block_size );
			auto i = index::Idx{}
					   .set_x( idx.x )
					   .set_y( idx.y )
					   .set_z( idx.z );
			auto reader = _->_.extract( i );
			_->decomp->transfer( reader, writer );
			return VolumeBlock<Voxel>( std::move( buffer ), _->block_dim );
		}
		uint3 index() const { return idx; }

	private:
		shared_ptr<VolumeInner<Voxel>> _;
		size_t offset;
		uint3 idx;
		friend struct Volume<Voxel>;
	};

	template <typename Voxel>
	struct Volume
	{
		static Volume from_compressed( const string &file_name )
		{
			Volume vol;
			ifstream is( file_name, ios::binary | ios::ate );
			auto is_len = is.tellg();
			auto p1 = file_name.find_last_of( '.' );
			auto p2 = file_name.find_last_of( '.', p1 - 1 );
			auto method = file_name.substr( p2 + 1, p1 - p2 - 1 );

			vol._ = make_shared<VolumeInner<Voxel>>( std::move( is ), is_len, method );
			auto len = vol._->_.block_size();
			vol.block_stride = vol._->block_size;
			vol.grid_dim = dim3( unsigned( vol._->_.adjusted().x / len ),
								 unsigned( vol._->_.adjusted().y / len ),
								 unsigned( vol._->_.adjusted().z / len ) );
			vol.all = vol.grid_dim * ( len - vol._->_.padding() * 2 );
			vol.raw_all = dim3{ unsigned( vol._->_.raw().x ),
								unsigned( vol._->_.raw().y ),
								unsigned( vol._->_.raw().z ) };
			return vol;
		}

	public:
		dim3 dim() const { return grid_dim; }
		cuda::Extent block_dim() const { return _->block_dim; }
		uint3 padding() const { return uint3{ unsigned( _->_.padding() ),
											  unsigned( _->_.padding() ),
											  unsigned( _->_.padding() ) }; }

		ArchievedVolumeBlock<Voxel> get_block( uint3 idx )
		{
			auto block_id = idx.x +
							idx.y * grid_dim.x +
							idx.z * grid_dim.x * grid_dim.y;
			ArchievedVolumeBlock<Voxel> block;
			block._ = _;
			block.offset = offset + block_id * block_stride + block_offset;
			block.idx = idx;
			return block;
		}
		vector<ArchievedVolumeBlock<Voxel>> get_blocks( uint3 idmin, uint3 idmax )
		{
			uint3 idx;
			vector<ArchievedVolumeBlock<Voxel>> _;
			for ( idx.z = idmin.z; idx.z < idmax.z; ++idx.z ) {
				for ( idx.y = idmin.y; idx.y < idmax.y; ++idx.y ) {
					for ( idx.x = idmin.x; idx.x < idmax.x; ++idx.x ) {
						_.emplace_back( get_block( idx ) );
					}
				}
			}
			return _;
		}
		vector<ArchievedVolumeBlock<Voxel>> get_blocks()
		{
			return get_blocks( uint3{ 0, 0, 0 }, uint3{ grid_dim.x, grid_dim.y, grid_dim.z } );
		}
		float scale() const
		{
			return min( min( float( all.x ) / raw_all.x,
							 float( all.y ) / raw_all.y ),
						float( all.z ) / raw_all.z );
		}

	private:
		shared_ptr<VolumeInner<Voxel>> _;
		dim3 grid_dim = dim3( 1, 1, 1 );
		size_t offset = 0;
		size_t block_stride = 0;
		size_t block_offset = 0;
		dim3 all, raw_all;
	};
}

VM_END_MODULE()
